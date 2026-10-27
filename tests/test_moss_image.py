"""Weight-free tests for the MOSS-Image port.

The torch modules below are a minimal copy of the reference pipeline
(OpenMOSS-Team/MOSS-Image-SFT, pipeline_moss_image.py) and of transformers'
Qwen3.5 text model. The MLX modules run on the same random weights at toy sizes.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_unflatten
from torch import nn as tnn
from torch.nn import functional as F

from mflux.models.common_models.qwen3_5 import Qwen35GatedDeltaNet, Qwen35TextModel
from mflux.models.moss_image.latent_creator.moss_image_latent_creator import MOSSImageLatentCreator
from mflux.models.moss_image.model.moss_image_scheduler.moss_image_scheduler import MOSSImageScheduler
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
from mflux.models.moss_image.variants.txt2img.moss_image import MOSSImage
from mflux.models.moss_image.weights.moss_image_weight_mapping import MOSSImageWeightMapping


class _TorchRMSNorm(tnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = tnn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _torch_situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _torch_attention(q, k, v, rope, heads):
    def rotate(x):
        sin, cos = rope.chunk(2, -1)
        sin = torch.cat((sin, sin), -1).unsqueeze(-2)
        cos = torch.cat((cos, cos), -1).unsqueeze(-2)
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat((-b, a), -1) * sin

    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape)).transpose(1, 2)
    k = rotate(k.reshape(shape)).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)


class _TorchTextBlock(tnn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _TorchRMSNorm(dim), _TorchRMSNorm(dim)
        self.q_norm, self.k_norm = _TorchRMSNorm(dim // heads), _TorchRMSNorm(dim // heads)
        self.qkv = tnn.Linear(dim, 3 * dim)
        self.proj = tnn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = tnn.Linear(dim, ffn_dim), tnn.Linear(dim, ffn_dim)
        self.ffn_down = tnn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_torch_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_torch_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TorchTextProjection(tnn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = tnn.Linear(text_dim, dim)
        self.blocks = tnn.ModuleList([_TorchTextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rope)
        return x


class _TorchBlock(tnn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _TorchRMSNorm(dim), _TorchRMSNorm(dim)
        self.q_norm, self.k_norm = _TorchRMSNorm(dim // heads), _TorchRMSNorm(dim // heads)
        self.proj = tnn.ModuleDict({name: tnn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = tnn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = tnn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_torch_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](_torch_situ(hidden, gate)) * gate_f[:, None]


class _TorchTransformer(tnn.Module):
    def __init__(self, dim, ffn_dim, num_heads, num_layers, text_dim, freq_dim, latent_channels, rope_base=10000.0):
        super().__init__()
        self.dim, self.freq_dim, self.num_bands = dim, freq_dim, dim // num_heads // 8
        self.rope_base = rope_base
        self.video_embedder = tnn.Linear(latent_channels, dim)
        self.text_embedder = _TorchTextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = tnn.Sequential(tnn.Linear(freq_dim, dim), tnn.SiLU(), tnn.Linear(dim, dim))
        self.shared_block_modulation = tnn.Linear(dim, 6 * dim)
        self.layers = tnn.ModuleList([_TorchBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _TorchRMSNorm(dim)
        self.final_linear_video = tnn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = tnn.Linear(dim, 2 * dim)

    def rope(self, height, width, text_length):
        bands = 1 / (self.rope_base ** (torch.arange(self.num_bands, dtype=torch.float32) / self.num_bands))
        h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4)
        text[:, 1] = torch.arange(text_length)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)

    def forward(self, hidden_states, timestep, encoder_hidden_states, skip_layers=()):
        _, _, height, width = hidden_states.shape
        half = self.freq_dim // 2
        phase = torch.tensor([timestep], dtype=torch.float64)[:, None] * 10000 ** (
            -torch.arange(half, dtype=torch.float64) / half
        )
        time = self.time_embedding(torch.cat((phase.cos(), phase.sin()), -1).float())
        mod = self.shared_block_modulation(time).view(1, 6, -1)
        head = self.video_head_modulation(time).view(1, 2, -1)
        video = self.video_embedder(hidden_states.flatten(2).transpose(1, 2))
        rope = self.rope(height, width, encoder_hidden_states.shape[1])
        x = torch.cat((video, self.text_embedder(encoder_hidden_states, rope[:, height * width :])), 1)
        for index, layer in enumerate(self.layers):
            if index not in skip_layers:
                x = layer(x, rope, mod)
        x = self.final_norm_video(x[:, : height * width])
        x = self.final_linear_video(x * (1 + head[:, 1, None]) + head[:, 0, None])
        return x.transpose(1, 2).reshape(1, -1, height, width)


class _Weights:
    @staticmethod
    def randomize(module: tnn.Module, seed: int = 0) -> dict[str, mx.array]:
        torch.manual_seed(seed)
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                parameter.normal_(0, 0.3 if "norm" in name else 0.08)
        return {k: mx.array(v.detach().numpy()) for k, v in module.state_dict().items()}


TOY_TRANSFORMER = dict(dim=96, ffn_dim=160, num_heads=4, num_layers=5, text_dim=48, freq_dim=32, latent_channels=128)


@pytest.mark.fast
@pytest.mark.parametrize("skip_layers", [(), (2,)])
def test_the_transformer_matches_the_reference_on_shared_weights(skip_layers):
    reference = _TorchTransformer(**TOY_TRANSFORMER).eval()
    model = MOSSImageTransformer(**TOY_TRANSFORMER)
    model.update(tree_unflatten(list(_Weights.randomize(reference).items())), strict=True)

    rng = np.random.default_rng(0)
    latents = rng.standard_normal((1, 128, 6, 4)).astype(np.float32)
    text = rng.standard_normal((1, 9, 48)).astype(np.float32)
    with torch.no_grad():
        expected = reference(torch.tensor(latents), 731.5, torch.tensor(text), skip_layers).numpy()
    actual = np.array(model(mx.array(latents), 731.5, mx.array(text), skip_layers=frozenset(skip_layers)))

    assert actual.shape == (1, 128, 6, 4)
    assert np.abs(actual - expected).max() < 1e-4


@pytest.mark.fast
def test_the_qwen3_5_text_model_matches_transformers_on_shared_weights():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    sizes = dict(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    rope = {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 2]}
    layer_types = Qwen35TextModel.layer_types_for(8, 4)
    reference = Qwen3_5TextModel(Qwen3_5TextConfig(**sizes, layer_types=layer_types, rope_parameters=rope)).eval()
    model = Qwen35TextModel(**sizes, full_attention_interval=4, partial_rotary_factor=0.25, rope_theta=1e7)
    model.update(tree_unflatten(list(_Weights.randomize(reference).items())), strict=True)

    # Longer than the reference's 64-token chunk, so the chunked and the sequential delta rule both run.
    input_ids = np.random.default_rng(0).integers(0, 97, (1, 150))
    with torch.no_grad():
        expected = reference(input_ids=torch.tensor(input_ids), use_cache=False).last_hidden_state.numpy()
    actual = np.array(model(mx.array(input_ids)))

    assert np.abs(actual - expected).max() < 1e-3


class _ReferenceSchedule:
    @staticmethod
    def sigmas(num_inference_steps: int) -> np.ndarray:
        def grid(count, shift):
            values = torch.linspace(1, shift / (1000 + shift - 1), count)
            return shift * values / (1 + (shift - 1) * values)

        sigmas, training = grid(num_inference_steps, 5.0), grid(1000, 1.0).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=999)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
        return torch.unique_consecutive(training[nearest]).numpy()


@pytest.mark.fast
@pytest.mark.parametrize("steps", [1, 4, 12, 28, 50, 100, 218])
def test_the_schedule_matches_the_reference(steps):
    sigmas = np.array(MOSSImageScheduler.compute_sigmas(steps))
    assert sigmas.shape == (steps + 1,)
    assert sigmas[-1] == 0.0
    assert np.abs(sigmas[:-1] - _ReferenceSchedule.sigmas(steps)).max() < 1e-6


@pytest.mark.fast
def test_the_schedule_refuses_a_step_count_the_training_grid_cannot_hold():
    with pytest.raises(ValueError, match="cannot run 219 steps"):
        MOSSImageScheduler.compute_sigmas(219)


@pytest.mark.fast
def test_the_euler_step_runs_in_float32():
    class _Config:
        num_inference_steps = 4

    scheduler = MOSSImageScheduler(_Config())
    latents = mx.ones((1, 128, 2, 2), dtype=mx.bfloat16)
    stepped = scheduler.step(noise=mx.ones((1, 128, 2, 2), dtype=mx.bfloat16), timestep=0, latents=latents)
    assert stepped.dtype == mx.float32
    expected = 1.0 + float(scheduler.sigmas[1] - scheduler.sigmas[0])
    assert np.allclose(np.array(stepped), expected)
    assert np.allclose(np.array(scheduler.timesteps), np.array(scheduler.sigmas[:-1]) * 1000)


@pytest.mark.fast
@pytest.mark.parametrize(("width", "height"), [(1000, 1024), (1024, 1000), (0, 1024)])
def test_the_noise_refuses_a_size_that_is_not_a_multiple_of_16(width, height):
    with pytest.raises(ValueError, match="multiples of 16"):
        MOSSImageLatentCreator.create_noise(seed=1, height=height, width=width)


@pytest.mark.fast
def test_the_noise_is_float32_at_one_sixteenth_of_the_image_size():
    noise = MOSSImageLatentCreator.create_noise(seed=1, height=320, width=192)
    assert noise.shape == (1, 128, 20, 12)
    assert noise.dtype == mx.float32


@pytest.mark.fast
@pytest.mark.parametrize("layers", [(42,), (-1,), (2.0,)])
def test_generation_refuses_a_skip_layer_outside_the_transformer(layers):
    model = MOSSImage.__new__(MOSSImage)
    model.transformer = type("Stub", (), {"layers": [None] * 42})()
    with pytest.raises(ValueError, match=r"layer indices in \[0, 42\)"):
        model._validate_skip_layers(layers)


@pytest.mark.fast
def test_the_mapping_lists_every_layer_of_the_published_model():
    transformer = {t.to_pattern for t in MOSSImageWeightMapping.get_transformer_mapping()}
    assert "layers.41.proj.ffn_down.weight" in transformer
    assert "layers.42.proj.ffn_down.weight" not in transformer
    assert "text_embedder.blocks.1.qkv.bias" in transformer

    text_encoder = {t.to_pattern for t in MOSSImageWeightMapping.get_text_encoder_mapping()}
    # Every fourth layer is full attention; the others are Gated DeltaNet layers.
    assert "layers.3.self_attn.q_proj.weight" in text_encoder
    assert "layers.3.linear_attn.A_log" not in text_encoder
    assert "layers.31.self_attn.o_proj.weight" in text_encoder
    assert "layers.30.linear_attn.conv1d.weight" in text_encoder
    assert all(t.required for t in MOSSImageWeightMapping.get_text_encoder_mapping())


@pytest.mark.fast
def test_the_step_cache_signal_is_one_timestep_embedding_per_sigma():
    model = MOSSImageTransformer(**TOY_TRANSFORMER)
    signal = model.time_signal(mx.array([1.0, 0.5, 0.25]))
    assert signal.shape == (3, TOY_TRANSFORMER["dim"])
    # The signal row is the embedding the forward pass modulates with at that timestep.
    single = model.time_signal(mx.array([0.5]))
    assert np.allclose(np.array(signal[1]), np.array(single[0]), atol=1e-6)


class _SequentialDeltaRule:
    # The token-by-token recurrence (transformers' torch_recurrent_gated_delta_rule), in numpy.
    @staticmethod
    def run(query, key, value, g, beta):
        def l2(x):
            return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)

        query, key = l2(query) * query.shape[-1] ** -0.5, l2(key)
        batch, seq_len, heads, key_dim = key.shape
        state = np.zeros((batch, heads, key_dim, value.shape[-1]))
        out = np.zeros(value.shape)
        for t in range(seq_len):
            state = state * np.exp(g[:, t])[:, :, None, None]
            kv_mem = (state * key[:, t][..., None]).sum(-2)
            delta = (value[:, t] - kv_mem) * beta[:, t][..., None]
            state = state + key[:, t][..., None] * delta[..., None, :]
            out[:, t] = (state * query[:, t][..., None]).sum(-2)
        return out


@pytest.mark.fast
def test_the_chunked_delta_rule_matches_the_recurrence_on_a_long_prompt():
    # 1000 tokens: 16 chunks, the last one padded. The chunked rule keeps the graph at one
    # serial step for each chunk, so the 4096-token prompt limit stays practical.
    rng = np.random.default_rng(0)
    shape = (1, 1000, 2, 16)
    query, key, value = (rng.standard_normal(shape) for _ in range(3))
    g = -np.abs(rng.standard_normal((1, 1000, 2))) * 0.1
    beta = 1 / (1 + np.exp(-rng.standard_normal((1, 1000, 2))))

    expected = _SequentialDeltaRule.run(query, key, value, g, beta)
    arrays = (mx.array(x.astype(np.float32)) for x in (query, key, value, g, beta))
    actual = np.array(Qwen35GatedDeltaNet._gated_delta_rule(*arrays))

    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 1e-3


@pytest.mark.fast
def test_the_chunked_delta_rule_survives_correlated_keys():
    # Deep Qwen3.5 DeltaNet layers have key cosines of 0.8 to 0.98 in a chunk. I.i.d. keys do not (issue #920).
    rng = np.random.default_rng(0)
    shape = (1, 1024, 2, 128)
    base = rng.standard_normal((1, 1, 2, 128))
    query = base + 0.3 * rng.standard_normal(shape)
    key = base + 0.3 * rng.standard_normal(shape)
    value = rng.standard_normal(shape)
    g = -np.abs(rng.standard_normal((1, 1024, 2))) * 0.02
    beta = 1 / (1 + np.exp(-(2.0 + rng.standard_normal((1, 1024, 2)))))

    expected = _SequentialDeltaRule.run(query, key, value, g, beta)
    arrays = (mx.array(x.astype(np.float32)) for x in (query, key, value, g, beta))
    actual = np.array(Qwen35GatedDeltaNet._gated_delta_rule(*arrays))

    assert np.isfinite(actual).all()
    # The GPU error is approximately 2e-3 (5e-7 on the CPU). The product form gives NaN.
    assert np.abs(actual - expected).max() < 1e-2
