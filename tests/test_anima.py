"""Weight-free tests for the Anima port.

The torch modules below are a minimal copy of the reference: ComfyUI's comfy/ldm/cosmos/predict2.py (MiniTrainDIT,
images only, AdaLN-LoRA, padding-mask channel) and comfy/ldm/anima/model.py (LLMAdapter), with the split-half RoPE
of comfy_kitchen. The MLX modules run on the same random weights at toy sizes. The sampler fixture was written with
ComfyUI's own sample_euler / sample_er_sde and "simple" scheduler (shift 3, multiplier 1).
"""

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx import nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from torch import nn as tnn
from torch.nn import functional as F

from mflux.models.anima.anima_initializer import AnimaInitializer
from mflux.models.anima.model.anima_scheduler import AnimaSampler, AnimaScheduler
from mflux.models.anima.model.anima_text_encoder.anima_prompt_encoder import AnimaPromptEncoder
from mflux.models.anima.model.anima_transformer.anima_llm_adapter import AnimaLLMAdapter
from mflux.models.anima.model.anima_transformer.anima_transformer import AnimaTransformer
from mflux.models.anima.weights.anima_lora_mapping import AnimaLoRAMapping
from mflux.models.anima.weights.anima_weight_definition import AnimaWeightDefinition
from mflux.models.anima.weights.anima_weight_mapping import AnimaWeightMapping
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer.tokenizer_output import TokenizerOutput
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

FIXTURE = Path(__file__).parent / "resources" / "anima_sampler_reference.npz"


def _rope_split_half(t, freqs):
    # comfy_kitchen apply_rope_split_half: pair k rotates elements k and k + head_dim / 2 by the 2x2 matrix freqs[k].
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs.dtype)
    out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    return out.movedim(-1, -2).reshape(*t.shape).type_as(t)


def _rope3d(head_dim, height, width, spatial_ratio=4.0):
    # VideoRopePosition3DEmb.generate_embeddings for one frame, as (L, head_dim / 2, 2, 2).
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    spatial_range = torch.arange(0, dim_h, 2)[: dim_h // 2].float() / dim_h
    temporal_range = torch.arange(0, dim_t, 2)[: dim_t // 2].float() / dim_t
    spatial = 1.0 / ((10000.0 * spatial_ratio ** (dim_h / (dim_h - 2))) ** spatial_range)
    temporal = 1.0 / (10000.0**temporal_range)
    seq = torch.arange(max(height, width), dtype=torch.float)

    def matrices(angles):
        return torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)], dim=-1)

    half_h = matrices(torch.outer(seq[:height], spatial))[:, None].expand(height, width, -1, 4)
    half_w = matrices(torch.outer(seq[:width], spatial))[None].expand(height, width, -1, 4)
    half_t = matrices(torch.outer(seq[:1], temporal)).expand(height, width, -1, 4)
    return torch.cat([half_t, half_h, half_w], dim=-2).reshape(height * width, head_dim // 2, 2, 2)


class _RefAttention(tnn.Module):
    def __init__(self, query_dim, context_dim, heads):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        self.heads = heads
        self.q_proj = tnn.Linear(query_dim, query_dim, bias=False)
        self.q_norm = tnn.RMSNorm(query_dim // heads, eps=1e-6)
        self.k_proj = tnn.Linear(context_dim, query_dim, bias=False)
        self.k_norm = tnn.RMSNorm(query_dim // heads, eps=1e-6)
        self.v_proj = tnn.Linear(context_dim, query_dim, bias=False)
        self.output_proj = tnn.Linear(query_dim, query_dim, bias=False)

    def forward(self, x, context=None, rope=None):
        context = x if context is None else context
        q = self.q_norm(self.q_proj(x).unflatten(-1, (self.heads, -1)))
        k = self.k_norm(self.k_proj(context).unflatten(-1, (self.heads, -1)))
        v = self.v_proj(context).unflatten(-1, (self.heads, -1))
        if self.is_selfattn and rope is not None:
            q, k = _rope_split_half(q, rope), _rope_split_half(k, rope)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.output_proj(out.transpose(1, 2).flatten(2))


def _ref_modulation(dim, lora_dim, chunks):
    return tnn.Sequential(tnn.SiLU(), tnn.Linear(dim, lora_dim, bias=False), tnn.Linear(lora_dim, chunks * dim, bias=False))  # fmt: off


class _RefBlock(tnn.Module):
    def __init__(self, dim, context_dim, heads, lora_dim):
        super().__init__()
        self.self_attn = _RefAttention(dim, None, heads)
        self.cross_attn = _RefAttention(dim, context_dim, heads)
        self.mlp = tnn.Module()
        self.mlp.layer1 = tnn.Linear(dim, 4 * dim, bias=False)
        self.mlp.layer2 = tnn.Linear(4 * dim, dim, bias=False)
        self.adaln_modulation_self_attn = _ref_modulation(dim, lora_dim, 3)
        self.adaln_modulation_cross_attn = _ref_modulation(dim, lora_dim, 3)
        self.adaln_modulation_mlp = _ref_modulation(dim, lora_dim, 3)

    def forward(self, x, emb, context, rope, adaln_lora):
        def norm(y):
            return F.layer_norm(y, (y.shape[-1],), eps=1e-6)

        shift, scale, gate = (self.adaln_modulation_self_attn(emb) + adaln_lora).chunk(3, dim=-1)
        x = x + gate * self.self_attn(norm(x) * (1 + scale) + shift, None, rope)
        shift, scale, gate = (self.adaln_modulation_cross_attn(emb) + adaln_lora).chunk(3, dim=-1)
        x = x + gate * self.cross_attn(norm(x) * (1 + scale) + shift, context)
        shift, scale, gate = (self.adaln_modulation_mlp(emb) + adaln_lora).chunk(3, dim=-1)
        return x + gate * self.mlp.layer2(F.gelu(self.mlp.layer1(norm(x) * (1 + scale) + shift)))


class _RefDiT(tnn.Module):
    # MiniTrainDIT for images: concat_padding_mask, use_adaln_lora, rope3d, patch (1, 2, 2), no extra pos emb.
    def __init__(self, dim, blocks, heads, context_dim, lora_dim, channels=16, patch=2):
        super().__init__()
        self.dim, self.heads, self.patch, self.channels = dim, heads, patch, channels
        self.x_embedder = tnn.Module()
        self.x_embedder.proj = tnn.Sequential(tnn.Identity(), tnn.Linear((channels + 1) * patch * patch, dim, bias=False))  # fmt: off
        self.t_embedder = tnn.Sequential(tnn.Identity(), tnn.Module())
        self.t_embedder[1].linear_1 = tnn.Linear(dim, dim, bias=False)
        self.t_embedder[1].linear_2 = tnn.Linear(dim, 3 * dim, bias=False)
        self.t_embedding_norm = tnn.RMSNorm(dim, eps=1e-6)
        self.blocks = tnn.ModuleList([_RefBlock(dim, context_dim, heads, lora_dim) for _ in range(blocks)])
        self.final_layer = tnn.Module()
        self.final_layer.adaln_modulation = _ref_modulation(dim, lora_dim, 2)
        self.final_layer.linear = tnn.Linear(dim, patch * patch * channels, bias=False)

    def forward(self, x, timesteps, context):
        b, c, t, h, w = x.shape
        p = self.patch
        x = torch.cat([x, torch.zeros(b, 1, t, h, w)], dim=1)
        # "b c (t r) (h m) (w n) -> b t h w (c r m n)"
        x = x.reshape(b, c + 1, t, h // p, p, w // p, p).permute(0, 2, 3, 5, 1, 4, 6).reshape(b, -1, (c + 1) * p * p)
        x = self.x_embedder.proj[1](x)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        phase = timesteps[:, None, None].float() * freqs
        sample = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        adaln_lora = self.t_embedder[1].linear_2(F.silu(self.t_embedder[1].linear_1(sample)))
        emb = self.t_embedding_norm(sample)
        rope = _rope3d(self.dim // self.heads, h // p, w // p)[None, :, None]
        for block in self.blocks:
            x = block(x, emb, context, rope, adaln_lora)
        shift, scale = (self.final_layer.adaln_modulation(emb) + adaln_lora[..., : 2 * self.dim]).chunk(2, dim=-1)
        x = self.final_layer.linear(F.layer_norm(x, (self.dim,), eps=1e-6) * (1 + scale) + shift)
        # "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)"
        x = x.reshape(b, t, h // p, w // p, p, p, 1, c).permute(0, 7, 1, 6, 2, 4, 3, 5)
        return x.reshape(b, c, t, h, w)


class _RefAdapterAttention(tnn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj = tnn.Linear(dim, dim, bias=False), tnn.Linear(dim, dim, bias=False)
        self.v_proj, self.o_proj = tnn.Linear(dim, dim, bias=False), tnn.Linear(dim, dim, bias=False)
        self.q_norm, self.k_norm = tnn.RMSNorm(dim // heads, eps=1e-6), tnn.RMSNorm(dim // heads, eps=1e-6)

    @staticmethod
    def rotary(length, head_dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        freqs = torch.outer(torch.arange(length).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    @staticmethod
    def rotate(x, cos, sin):
        half = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin

    def forward(self, x, context):
        q = self.q_norm(self.q_proj(x).unflatten(-1, (self.heads, -1))).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).unflatten(-1, (self.heads, -1))).transpose(1, 2)
        v = self.v_proj(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q = self.rotate(q, *self.rotary(q.shape[2], q.shape[-1]))
        k = self.rotate(k, *self.rotary(k.shape[2], k.shape[-1]))
        return self.o_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))


class _RefAdapterBlock(tnn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm_self_attn, self.self_attn = tnn.RMSNorm(dim, eps=1e-6), _RefAdapterAttention(dim, heads)
        self.norm_cross_attn, self.cross_attn = tnn.RMSNorm(dim, eps=1e-6), _RefAdapterAttention(dim, heads)
        self.norm_mlp = tnn.RMSNorm(dim, eps=1e-6)
        self.mlp = tnn.Sequential(tnn.Linear(dim, 4 * dim), tnn.GELU(), tnn.Linear(4 * dim, dim))

    def forward(self, x, context):
        normed = self.norm_self_attn(x)
        x = x + self.self_attn(normed, normed)
        x = x + self.cross_attn(self.norm_cross_attn(x), context)
        return x + self.mlp(self.norm_mlp(x))


class _RefAdapter(tnn.Module):
    def __init__(self, vocab, dim, layers, heads):
        super().__init__()
        self.embed = tnn.Embedding(vocab, dim)
        self.blocks = tnn.ModuleList([_RefAdapterBlock(dim, heads) for _ in range(layers)])
        self.out_proj = tnn.Linear(dim, dim)
        self.norm = tnn.RMSNorm(dim, eps=1e-6)

    def forward(self, source, ids):
        x = self.embed(ids).to(source.dtype)
        for block in self.blocks:
            x = block(x, source)
        return self.norm(self.out_proj(x))


def _load_into_mlx(module: nn.Module, reference: tnn.Module, skip_prefix: str | None = None) -> None:
    # Identity names: the reference's state dict is the checkpoint layout without its file prefix.
    weights = {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
    expected = {k for k, _ in tree_flatten(module.parameters()) if skip_prefix is None or not k.startswith(skip_prefix)}
    assert set(weights) == expected
    module.update(tree_unflatten(list(weights.items())))


def _randomize(reference: tnn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in reference.named_parameters():
            if param.ndim > 1:
                param.copy_(torch.randn_like(param) / math.sqrt(param.shape[-1]))
            elif name.endswith("bias"):
                param.copy_(0.1 * torch.randn_like(param))
            else:
                param.copy_(1.0 + 0.1 * torch.randn_like(param))


def _module_at(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


@pytest.mark.fast
class TestAnimaTransformer:
    def test_the_transformer_matches_the_reference(self):
        reference = _RefDiT(dim=128, blocks=2, heads=2, context_dim=64, lora_dim=32)
        _randomize(reference, seed=0)
        model = AnimaTransformer(dim=128, num_blocks=2, num_heads=2, context_dim=64, adaln_lora_dim=32)
        _load_into_mlx(model, reference, skip_prefix="llm_adapter.")
        torch.manual_seed(1)
        x = torch.randn(1, 16, 1, 12, 8)
        sigma = torch.tensor([0.7])
        context = torch.randn(1, 20, 64)

        with torch.no_grad():
            expected = reference(x, sigma, context)[:, :, 0].numpy()
        out = model(mx.array(x[:, :, 0].numpy()), mx.array(sigma.numpy()), mx.array(context.numpy()))

        np.testing.assert_allclose(np.array(out), expected, rtol=2e-4, atol=2e-4)

    def test_float16_compute_casts_the_blocks_only_and_keeps_the_output_close(self):
        model = AnimaTransformer(dim=128, num_blocks=2, num_heads=2, context_dim=64, adaln_lora_dim=32)
        model.update(tree_map(lambda v: v.astype(mx.bfloat16), model.parameters()))
        mx.random.seed(3)
        args = (mx.random.normal((1, 16, 12, 8)), mx.array([0.7]), mx.random.normal((1, 20, 64)).astype(mx.bfloat16))
        reference = model(*args)

        model.apply_compute_precision(ComputePrecision(mx.float16))
        out = model(*args)

        block = model.blocks[0]
        for linear in (block.self_attn.q_proj, block.cross_attn.v_proj, block.mlp.layer2):
            assert linear.weight.dtype == mx.float16
        assert block.adaln_modulation_mlp[1].weight.dtype == mx.bfloat16
        assert {v.dtype for _, v in tree_flatten(model.llm_adapter.parameters())} == {mx.bfloat16}
        assert out.dtype == reference.dtype
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)), np.array(reference.astype(mx.float32)), atol=5e-2, rtol=5e-2
        )

    def test_the_llm_adapter_matches_the_reference(self):
        reference = _RefAdapter(vocab=96, dim=64, layers=2, heads=4)
        _randomize(reference, seed=2)
        adapter = AnimaLLMAdapter(vocab_size=96, dim=64, num_layers=2, num_heads=4)
        _load_into_mlx(adapter, reference)
        torch.manual_seed(3)
        source = torch.randn(1, 9, 64)
        ids = torch.randint(0, 96, (1, 7))

        with torch.no_grad():
            expected = reference(source, ids).numpy()
        out = adapter(mx.array(source.numpy()), mx.array(ids.numpy()))

        np.testing.assert_allclose(np.array(out), expected, rtol=2e-4, atol=2e-4)

    def test_the_context_is_zero_padded_to_512_tokens_and_takes_the_blocks_dtype(self):
        model = AnimaTransformer(dim=128, num_blocks=1, num_heads=2, context_dim=64, adaln_lora_dim=32)
        model.update(tree_unflatten([(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())]))

        context = model.encode_text(mx.ones((1, 5, 64), dtype=mx.float32), mx.array([[3, 4, 1]]))

        assert context.shape == (1, 512, 64)
        assert context.dtype == mx.bfloat16
        assert float(mx.abs(context[:, 3:]).max()) == 0.0

    def test_a_context_longer_than_512_tokens_is_kept_whole(self):
        model = AnimaTransformer(dim=128, num_blocks=1, num_heads=2, context_dim=64, adaln_lora_dim=32)

        context = model.encode_text(mx.ones((1, 5, 64)), mx.zeros((1, 600), dtype=mx.int32))

        assert context.shape == (1, 600, 64)


class _FakeTokenizer:
    def __init__(self, ids):
        self.ids = ids

    def tokenize(self, prompt, keep_empty=False, **kwargs):
        ids = mx.array(np.array([self.ids], dtype=np.int32).reshape(1, len(self.ids)))
        return TokenizerOutput(input_ids=ids, attention_mask=mx.ones_like(ids))


class _RecordingEncoder:
    def __init__(self):
        self.seen = None

    def __call__(self, input_ids, dtype=None):
        self.seen = (np.array(input_ids).tolist(), dtype)
        return mx.zeros((1, input_ids.shape[1], 64)), None


@pytest.mark.fast
class TestAnimaPromptEncoder:
    def test_an_empty_prompt_reads_as_the_qwen_pad_token_and_the_text_path_runs_in_float32(self):
        encoder = _RecordingEncoder()
        transformer = AnimaTransformer(dim=128, num_blocks=1, num_heads=2, context_dim=64, adaln_lora_dim=32)

        context = AnimaPromptEncoder.encode("", _FakeTokenizer([]), _FakeTokenizer([1]), encoder, transformer)

        assert encoder.seen == ([[AnimaPromptEncoder.QWEN_PAD_TOKEN]], mx.float32)
        assert context.shape == (1, 512, 64)


@pytest.mark.fast
class TestAnimaSampling:
    @pytest.mark.parametrize("steps", [1, 8, 10, 12, 30, 50])
    def test_the_sigmas_are_comfyuis_simple_schedule(self, steps):
        expected = np.load(FIXTURE)[f"sigmas_{steps}"]

        np.testing.assert_array_equal(np.array(AnimaScheduler.compute_sigmas(steps)), expected)

    @pytest.mark.parametrize("name", ["euler", "er_sde"])
    @pytest.mark.parametrize("steps", [8, 30])
    def test_the_samplers_follow_comfyui(self, name, steps):
        fixture = np.load(FIXTURE)
        sampler = AnimaSampler.make(name, AnimaScheduler.compute_sigmas(steps), seed=0)
        if name == "er_sde":
            noise = [mx.array(n) for n in fixture[f"er_sde_{steps}_noise"]]
            sampler._noise = lambda shape: noise.pop(0)
        x = mx.array(fixture["x0"])

        for t in range(steps):
            sigma = sampler.sigma(t)
            denoised = 0.6 * x * (1 - sigma) + 0.1 * mx.tanh(x)
            x = sampler.step(t, x, (x - denoised) / sigma, denoised)

        np.testing.assert_allclose(np.array(x), fixture[f"{name}_{steps}"], rtol=1e-5, atol=1e-5)

    def test_er_sde_evaluates_the_first_step_just_below_sigma_one(self):
        sampler = AnimaSampler.make("er_sde", AnimaScheduler.compute_sigmas(10), seed=0)

        assert float(sampler.sigma(0)) == pytest.approx(3 * 0.9999 / (1 + 2 * 0.9999))

    def test_an_unknown_sampler_is_refused(self):
        with pytest.raises(ValueError, match="er_sde, euler"):
            AnimaSampler.make("dpmpp_2m", AnimaScheduler.compute_sigmas(4), seed=0)


@pytest.mark.fast
class TestAnimaWeights:
    def test_the_mapping_names_every_transformer_parameter(self):
        # Full depth (28 blocks, 6 adapter blocks) at toy widths: the names are those of the real checkpoint.
        model = AnimaTransformer(dim=64, num_blocks=28, num_heads=1, context_dim=64, adaln_lora_dim=32)
        names = {target.to_pattern for target in AnimaWeightMapping.get_transformer_mapping()}

        assert names == {key for key, _ in tree_flatten(model.parameters())}
        assert len(names) == 685

    @pytest.mark.parametrize("prefix", ["model.diffusion_model.", "net.", "diffusion_model.", ""])
    def test_the_transformer_keys_lose_their_file_prefix(self, prefix):
        assert AnimaWeightMapping.transformer_key(f"{prefix}blocks.3.self_attn.q_proj.weight") == "blocks.3.self_attn.q_proj.weight"  # fmt: off

    def test_the_comfyui_vae_keys_cover_the_qwen_image_vae_mapping(self):
        # ComfyUI's qwen_image_vae.safetensors in the Wan 2.1 layout, built from the architecture.
        keys = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]
        residual = ["residual.0.gamma", "residual.2.weight", "residual.2.bias", "residual.3.gamma", "residual.6.weight", "residual.6.bias"]  # fmt: off
        for side in ("encoder", "decoder"):
            keys += [f"{side}.conv1.weight", f"{side}.conv1.bias", f"{side}.head.0.gamma", f"{side}.head.2.weight", f"{side}.head.2.bias"]  # fmt: off
            keys += [f"{side}.middle.{i}.{r}" for i in (0, 2) for r in residual]
            keys += [f"{side}.middle.1.{n}" for n in ("norm.gamma", "to_qkv.weight", "to_qkv.bias", "proj.weight", "proj.bias")]  # fmt: off
        for i in range(11):
            if i in (2, 5, 8):
                keys += [f"encoder.downsamples.{i}.resample.1.weight", f"encoder.downsamples.{i}.resample.1.bias"]
                keys += [f"encoder.downsamples.{i}.time_conv.{p}" for p in ("weight", "bias") if i != 2]
            else:
                keys += [f"encoder.downsamples.{i}.{r}" for r in residual]
        keys += [f"encoder.downsamples.{i}.shortcut.{p}" for i in (3, 6) for p in ("weight", "bias")]
        for i in range(15):
            if i in (3, 7, 11):
                keys += [f"decoder.upsamples.{i}.resample.1.weight", f"decoder.upsamples.{i}.resample.1.bias"]
                keys += [f"decoder.upsamples.{i}.time_conv.{p}" for p in ("weight", "bias") if i != 11]
            else:
                keys += [f"decoder.upsamples.{i}.{r}" for r in residual]
        keys += [f"decoder.upsamples.4.shortcut.{p}" for p in ("weight", "bias")]
        assert len(keys) == 194
        converted = {QwenWeightMapping.vae_key_from_comfy(key): mx.zeros((1,)) for key in keys}

        mapping = QwenWeightMapping.get_vae_mapping()
        assert WeightMapper.missing_required_names(hf_weights=converted, mapping=mapping, num_blocks=None, num_layers=None) == []  # fmt: off
        assert QwenWeightMapping.vae_key_from_comfy("decoder.upsamples.12.residual.6.weight") == "decoder.up_blocks.3.resnets.0.conv2.weight"  # fmt: off
        assert QwenWeightMapping.vae_key_from_comfy("decoder.upsamples.7.time_conv.bias") == "decoder.up_blocks.1.upsamplers.0.time_conv.bias"  # fmt: off

    def test_a_repository_copy_reads_the_comfyui_files_and_a_save_its_own_layout(self, tmp_path):
        copy = tmp_path / "Anima"
        (copy / AnimaWeightDefinition.TRANSFORMER_DIR).mkdir(parents=True)
        (copy / AnimaWeightDefinition.TRANSFORMER_DIR / "anima-base-v1.0.safetensors").touch()
        saved = tmp_path / "Anima-base-mflux-q8"
        (saved / "transformer").mkdir(parents=True)

        on_copy = {c.name: c.variant_selector(copy) for c in AnimaWeightDefinition.get_components()}
        on_save = {c.name: c.variant_selector(saved) for c in AnimaWeightDefinition.get_components()}

        assert on_copy["transformer"].weight_files == ["anima-base-v1.0.safetensors"]
        assert on_copy["vae"].hf_subdir == "split_files/vae"
        assert on_copy["text_encoder"].skip_quantization
        assert {name: c.hf_subdir for name, c in on_save.items()} == {name: name for name in on_save}
        assert AnimaInitializer.is_upstream(AnimaWeightDefinition.REPO) and AnimaInitializer.is_upstream(str(copy))
        assert not AnimaInitializer.is_upstream(str(saved))

    def test_three_bit_quantization_fails_before_any_weight_loads(self):
        with pytest.raises(ValueError, match="-q 3"):
            AnimaInitializer.init(model=None, model_config=None, quantize=3)

    def test_quantization_keeps_the_text_path_the_modulations_and_the_patch_embedding_dense(self):
        model = AnimaTransformer(dim=128, num_blocks=1, num_heads=2, context_dim=64, adaln_lora_dim=64)

        nn.quantize(model, bits=4, class_predicate=AnimaWeightDefinition.quantization_predicate)

        quantized = {key.rsplit(".", 1)[0] for key, _ in tree_flatten(model.parameters()) if key.endswith(".scales")}
        assert "blocks.0.self_attn.q_proj" in quantized and "blocks.0.mlp.layer1" in quantized
        assert not any(name.startswith(("llm_adapter", "t_embedder", "x_embedder")) for name in quantized)
        assert not any("adaln_modulation" in name for name in quantized)
        assert {c.name for c in AnimaWeightDefinition.get_components() if c.skip_quantization} == {"text_encoder"}
        assert AnimaWeightDefinition.source_text_encoder().skip_quantization


@pytest.mark.fast
class TestAnimaLoRAMapping:
    @pytest.mark.parametrize(
        ("key", "target"),
        [
            ("lora_unet_blocks_7_self_attn_q_proj.lora_down.weight", "blocks.7.self_attn.q_proj"),
            ("lora_unet_blocks_27_mlp_layer2.lora_up.weight", "blocks.27.mlp.layer2"),
            ("lora_unet_blocks_0_cross_attn_output_proj.alpha", "blocks.0.cross_attn.output_proj"),
            ("diffusion_model.blocks.3.cross_attn.k_proj.lora_A.weight", "blocks.3.cross_attn.k_proj"),
            ("diffusion_model.blocks.3.adaln_modulation_mlp.2.lora_B.weight", "blocks.3.adaln_modulation_mlp.2"),
            (
                "lora_unet_llm_adapter_blocks_5_cross_attn_o_proj.lora_up.weight",
                "llm_adapter.blocks.5.cross_attn.o_proj",
            ),
            ("lycoris_blocks_1_self_attn_v_proj.lokr_w1", "blocks.1.self_attn.v_proj"),
        ],
    )
    def test_community_key_spellings_reach_their_module(self, key, target):
        found = []
        for t in AnimaLoRAMapping.get_mapping():
            patterns = t.possible_up_patterns + t.possible_down_patterns + t.possible_alpha_patterns
            for pattern in patterns + t.possible_lokr_w1_patterns:
                block = LoRALoader._match_pattern(key, pattern)
                if block is not None:
                    found.append(t.model_path.format(block=block))

        assert found == [target]

    def test_a_kohya_lora_applies_to_every_block_layer(self, tmp_path):
        # The layout of kohya's networks.lora_anima: ten layers per block, alpha beside each pair.
        model = AnimaTransformer(dim=128, num_blocks=2, num_heads=2, context_dim=64, adaln_lora_dim=32)
        layers = [f"{attn}.{proj}" for attn in ("self_attn", "cross_attn") for proj in ("q_proj", "k_proj", "v_proj", "output_proj")]  # fmt: off
        targets = [f"blocks.{block}.{layer}" for block in range(2) for layer in [*layers, "mlp.layer1", "mlp.layer2"]]
        weights = {}
        for path in targets:
            out_dim, in_dim = _module_at(model, path).weight.shape
            name = f"lora_unet_{path.replace('.', '_')}"
            weights[f"{name}.lora_down.weight"] = mx.full((4, in_dim), 0.01)
            weights[f"{name}.lora_up.weight"] = mx.full((out_dim, 4), 0.01)
            weights[f"{name}.alpha"] = mx.array(4.0)
        lora_file = tmp_path / "anima_style.safetensors"
        mx.save_safetensors(str(lora_file), weights)
        before = {path: _module_at(model, path).weight for path in targets}

        paths, scales = LoRALoader.load_and_apply_lora(AnimaLoRAMapping.get_mapping(), model, [str(lora_file)], [1.0])

        assert paths == [str(lora_file)] and scales == [1.0]
        for path, weight in before.items():
            assert not mx.array_equal(_module_at(model, path).weight, weight), path
