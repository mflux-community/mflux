import json
import struct
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten, tree_map
from torch import nn as tnn
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common_models.qwen3_5 import Qwen35GatedDeltaNet
from mflux.models.qwen21 import qwen21_initializer, qwen_image21_initializer
from mflux.models.qwen21.cli import qwen21_edit_generate, qwen21_generate
from mflux.models.qwen21.model.qwen21_small_text_encoder import zen_image_edit_weights
from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen21_small_text_encoder import Qwen21SmallTextEncoder
from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen21_text_fusion import Qwen21TextFusion
from mflux.models.qwen21.model.qwen21_small_text_encoder.zen_image_edit_weights import ZenImageEditWeights
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.models.qwen21.weights.qwen21_weight_definition import (
    Qwen21NoTextEncoderWeightDefinition,
    Qwen21WeightDefinition,
)
from mflux.models.qwen21.weights.qwen_image21_weight_definition import (
    QwenImage21NoTextEncoderWeightDefinition,
    QwenImage21WeightDefinition,
)

IMAGE_TOKEN, VISION_START, VISION_END = 90, 91, 92
HIDDEN, LAYERS, STUDENT_LAYERS = 32, 8, (2, 4, 6, 8)

TEXT_CONFIG = {
    "hidden_size": HIDDEN,
    "intermediate_size": 64,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
    "full_attention_interval": 4,
    "vocab_size": 100,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "attn_output_gate": True,
    "max_position_embeddings": 4096,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 100,  # a small theta makes every M-RoPE axis rotate visibly
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
    },
    "tie_word_embeddings": True,
}
VISION_CONFIG = {
    "depth": 2,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_heads": 2,
    "out_hidden_size": HIDDEN,
    "patch_size": 4,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
}
ENCODER_CONFIG = {
    "text_config": TEXT_CONFIG,
    "vision_config": VISION_CONFIG,
    "image_token_id": IMAGE_TOKEN,
    "vision_start_token_id": VISION_START,
    "vision_end_token_id": VISION_END,
    "tie_word_embeddings": True,
}
FUSION_CONFIG = {
    "in_dim": HIDDEN * len(STUDENT_LAYERS),
    "out_dim": 24,
    "hidden": 48,
    "proj_layers": 3,
    "norm": "ln-per-layer",
    "n_slices": len(STUDENT_LAYERS),
    "attention": 2,
    "attn_dim": 16,
    "attn_heads": 2,
    "max_len": 20,
    "mixer": 2,
    "mixer_heads": 2,
    "mixer_ffn": 2,
    "student_layers": list(STUDENT_LAYERS),
    "drop_idx": 3,
}


class TorchAttnBlock(tnn.Module):
    def __init__(self, dim, heads, ffn_mult):
        super().__init__()
        self.n1 = tnn.LayerNorm(dim)
        self.attn = tnn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = tnn.LayerNorm(dim)
        self.ffn = tnn.Sequential(tnn.Linear(dim, dim * ffn_mult), tnn.GELU(approximate="tanh"), tnn.Linear(dim * ffn_mult, dim))  # fmt: skip

    def forward(self, x, key_padding_mask=None):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)[0]
        return x + self.ffn(self.n2(x))


class TorchFusion(tnn.Module):
    # Independent torch statement of the zen-image-edit text_fusion forward pass.
    def __init__(self, c):
        super().__init__()
        self.n, self.d = c["n_slices"], c["in_dim"] // c["n_slices"]
        norm = tnn.Module()
        norm.all = tnn.LayerNorm(c["in_dim"])
        self.mlp = tnn.Sequential(
            norm,
            tnn.Linear(c["in_dim"], c["hidden"]),
            tnn.GELU(approximate="tanh"),
            tnn.Linear(c["hidden"], c["hidden"]),
            tnn.GELU(approximate="tanh"),
            tnn.Linear(c["hidden"], c["out_dim"]),
        )
        self.attn = tnn.Module()
        self.attn.norm = tnn.LayerNorm(c["in_dim"])
        self.attn.inp = tnn.Linear(c["in_dim"], c["attn_dim"])
        self.attn.pos = tnn.Parameter(torch.randn(1, c["max_len"], c["attn_dim"]))
        self.attn.blocks = tnn.ModuleList([TorchAttnBlock(c["attn_dim"], c["attn_heads"], 4) for _ in range(2)])
        self.attn.out = tnn.Linear(c["attn_dim"], c["out_dim"])
        self.mixer = tnn.Module()
        self.mixer.blocks = tnn.ModuleList([TorchAttnBlock(self.d, c["mixer_heads"], c["mixer_ffn"]) for _ in range(2)])
        self.mixer.proj = tnn.Linear(self.n, 1, bias=False)
        self.mixer.out = tnn.Linear(self.d, c["out_dim"])

    def per_slice(self, x):
        b, n, _ = x.shape
        return tnn.functional.layer_norm(x.view(b, n, self.n, self.d), (self.d,))

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.mlp[0].all(self.per_slice(x).reshape(b, n, -1))
        for layer in list(self.mlp)[1:]:
            h = layer(h)
        a = self.attn.inp(self.attn.norm(x))
        pos = tnn.functional.pad(self.attn.pos, (0, 0, 0, max(0, n - self.attn.pos.shape[1])))
        a = a + pos[:, :n]
        for block in self.attn.blocks:
            a = block(a, None if mask is None else ~mask)
        m = self.per_slice(x).reshape(b * n, self.n, self.d)
        for block in self.mixer.blocks:
            m = block(m)
        m = self.mixer.out(self.mixer.proj(m.permute(0, 2, 1)).squeeze(-1)).reshape(b, n, -1)
        return h + self.attn.out(a) + m


def _to_mx(state: dict) -> dict:
    return {key: mx.array(value.detach().float().numpy()) for key, value in state.items()}


def _reference_model() -> Qwen3_5ForConditionalGeneration:
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**ENCODER_CONFIG)).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight") and "visual" not in name:
                param.normal_(0, 0.3)  # zero-centred norms start at 0; make (1 + w) non-trivial
            elif name.endswith("A_log") or name.endswith("dt_bias"):
                param.uniform_(-1, 1)
    return model


def _mlx_encoder(reference: Qwen3_5ForConditionalGeneration) -> Qwen21SmallTextEncoder:
    encoder = Qwen21SmallTextEncoder(ENCODER_CONFIG, FUSION_CONFIG)
    encoder.load_encoder_weights(_to_mx(reference.state_dict()), language_model_dtype=mx.float32)
    return encoder


def _float32(module) -> None:
    module.update(tree_map(lambda v: v.astype(mx.float32), module.parameters()))


def _reference_slices(model, input_ids, **kwargs) -> np.ndarray:
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, **kwargs)
    states = torch.cat([out.hidden_states[i] for i in STUDENT_LAYERS], dim=-1)
    return states.float().numpy()


@pytest.mark.fast
def test_text_slices_match_transformers_across_several_delta_chunks():
    reference = _reference_model()
    encoder = _mlx_encoder(reference)
    _float32(encoder)
    ids = torch.randint(0, 89, (1, 150))  # three 64-token chunks of the gated delta rule
    expected = _reference_slices(reference, ids)
    actual = np.array(encoder.hidden_slices(mx.array(ids.numpy())))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-5)


@pytest.mark.fast
def test_image_slices_match_transformers_with_mrope_positions():
    reference = _reference_model()
    encoder = _mlx_encoder(reference)
    _float32(encoder)
    grid = torch.tensor([[1, 4, 6]])  # 24 patches -> 6 merged image tokens
    pixel_values = torch.randn(24, 3 * 2 * 4 * 4)
    ids = torch.cat(
        [
            torch.randint(0, 89, (1, 5)),
            torch.tensor([[VISION_START] + [IMAGE_TOKEN] * 6 + [VISION_END]]),
            torch.randint(0, 89, (1, 9)),
        ],
        dim=1,
    )
    mm_types = (ids == IMAGE_TOKEN).int()
    expected = _reference_slices(
        reference, ids, pixel_values=pixel_values, image_grid_thw=grid, mm_token_type_ids=mm_types
    )
    actual = np.array(
        encoder.hidden_slices(mx.array(ids.numpy()), mx.array(pixel_values.numpy()), mx.array(grid.numpy()))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-5)


@pytest.mark.fast
def test_chunked_delta_rule_matches_the_token_by_token_recurrence():
    rng = np.random.default_rng(0)
    q, k = (mx.array(rng.standard_normal((1, 70, 2, 8)).astype(np.float32)) for _ in range(2))
    v = mx.array(rng.standard_normal((1, 70, 2, 4)).astype(np.float32))
    decay = mx.array(-rng.uniform(0, 1, (1, 70, 2)).astype(np.float32))
    beta = mx.array(rng.uniform(0, 1, (1, 70, 2)).astype(np.float32))
    chunked = np.array(Qwen35GatedDeltaNet._gated_delta_rule(q, k, v, decay, beta, chunk=16))
    qn = np.array(Qwen35GatedDeltaNet._l2_normalize(q)) * 8**-0.5
    kn = np.array(Qwen35GatedDeltaNet._l2_normalize(k))
    state = np.zeros((2, 8, 4), dtype=np.float32)
    expected = np.zeros((1, 70, 2, 4), dtype=np.float32)
    for t in range(70):
        state *= np.exp(np.array(decay)[0, t])[:, None, None]
        delta = (np.array(v)[0, t] - np.einsum("hk,hkv->hv", kn[0, t], state)) * np.array(beta)[0, t][:, None]
        state += np.einsum("hk,hv->hkv", kn[0, t], delta)
        expected[0, t] = np.einsum("hk,hkv->hv", qn[0, t], state)
    np.testing.assert_allclose(chunked, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.fast
@pytest.mark.parametrize("padded", [False, True])
def test_text_fusion_matches_a_torch_statement_of_the_adapter(padded):
    torch.manual_seed(1)
    reference = TorchFusion(FUSION_CONFIG).eval()
    fusion = Qwen21TextFusion(FUSION_CONFIG)
    Qwen21SmallTextEncoder.load_fusion_weights(fusion, _to_mx(reference.state_dict()))
    length = 26  # past the 20-slot position table: the tail runs without positions
    x = torch.randn(1, length, FUSION_CONFIG["in_dim"]) * 5
    mask = torch.ones(1, length, dtype=torch.bool)
    if padded:
        mask[:, -4:] = False
    with torch.no_grad():
        expected = reference(x, mask if padded else None).numpy()
    actual = np.array(fusion(mx.array(x.numpy()), mx.array(mask.numpy()) if padded else None))
    rows = slice(None) if not padded else slice(0, length - 4)
    np.testing.assert_allclose(actual[:, rows], expected[:, rows], rtol=1e-4, atol=1e-4)


@pytest.mark.fast
def test_a_fusion_key_that_does_not_fit_is_named():
    raw = _to_mx(TorchFusion(FUSION_CONFIG).state_dict())
    del raw["mixer.out.bias"]
    with pytest.raises(ValueError, match="mixer.out.bias"):
        Qwen21SmallTextEncoder.load_fusion_weights(Qwen21TextFusion(FUSION_CONFIG), raw)


@pytest.mark.fast
def test_the_encoder_drops_mtp_weights_and_rejects_a_missing_tensor():
    raw = _to_mx(_reference_model().state_dict())
    raw["mtp.fc.weight"] = mx.zeros((2, 2))
    encoder = Qwen21SmallTextEncoder(ENCODER_CONFIG, FUSION_CONFIG)
    encoder.load_encoder_weights(raw)
    del raw["model.language_model.norm.weight"]
    with pytest.raises(ValueError, match="language_model.norm.weight"):
        encoder.load_encoder_weights(raw)


class FakeTokenizer:
    def __init__(self, prefix_tokens: int):
        self.prefix_tokens = prefix_tokens
        self.texts = []

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return {"input_ids": list(range(self.prefix_tokens))}
        self.texts.extend(text)
        return {"input_ids": np.array([[1, 2, 3, VISION_START, IMAGE_TOKEN, VISION_END, 4, 5, 6]])}


@pytest.mark.fast
def test_encode_uses_the_training_template_and_cuts_the_prefix_after_the_fusion():
    encoder = Qwen21SmallTextEncoder(ENCODER_CONFIG, FUSION_CONFIG)
    seen = {}

    def processor(text, **kwargs):
        seen["text"], seen["images"] = text[0], kwargs.get("images")
        return {
            "input_ids": np.array([[1, 2, 3, VISION_START] + [IMAGE_TOKEN] * 2 + [VISION_END, 4]]),
            "pixel_values": np.zeros((8, 3 * 2 * 4 * 4), dtype=np.float32),
            "image_grid_thw": np.array([[1, 2, 4]]),
        }

    from PIL import Image

    rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    embeds, slots = encoder.encode("make it blue", [rgba], processor)
    assert seen["text"] == (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>make it blue<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert seen["images"][0].mode == "RGB" and seen["images"][0].getpixel((0, 0)) == (255, 255, 255)
    assert embeds.shape == (1, 8 - FUSION_CONFIG["drop_idx"], FUSION_CONFIG["out_dim"])
    assert slots.tolist() == [False, True, True, False, False]


@pytest.mark.fast
def test_a_tokenizer_whose_system_prefix_disagrees_with_drop_idx_is_rejected():
    Qwen21SmallTextEncoder._check_drop_idx(SimpleNamespace(tokenizer=FakeTokenizer(14)), 14)
    with pytest.raises(ValueError, match="drop_idx is 14"):
        Qwen21SmallTextEncoder._check_drop_idx(SimpleNamespace(tokenizer=FakeTokenizer(13)), 14)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("full", "small"),
    [
        (Qwen21WeightDefinition, Qwen21NoTextEncoderWeightDefinition),
        (QwenImage21WeightDefinition, QwenImage21NoTextEncoderWeightDefinition),
    ],
)
def test_the_small_encoder_definitions_never_fetch_the_8b_text_encoder(full, small):
    assert [c.name for c in small.get_components()] == [
        c.name for c in full.get_components() if c.name != "text_encoder"
    ]
    assert not any(p.startswith("text_encoder/") for p in small.get_download_patterns())
    assert set(small.get_download_patterns()) < set(full.get_download_patterns())


def _safetensors(path, tensors: dict[str, np.ndarray]) -> None:
    header, offset, blobs = {}, 0, []
    for key, value in tensors.items():
        blob = value.astype(np.float16).tobytes()
        header[key] = {"dtype": "F16", "shape": list(value.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


class FakeRangeResponse:
    def __init__(self, status_code=206, content_range=None, chunks=None):
        self.status_code = status_code
        self._headers = {"content-range": content_range} if content_range is not None else {}
        self._chunks = list(chunks or [])

    @property
    def is_error(self):
        return 400 <= self.status_code < 600

    @property
    def headers(self):
        return self._headers

    def iter_bytes(self):
        return iter(self._chunks)

    def raise_for_status(self):
        if self.is_error:
            raise OSError(f"HTTP {self.status_code}")
        return self


def _range_cm(response):
    @contextmanager
    def cm():
        yield response

    return cm()


class FakeRangeFileSystem:
    class _Api:
        def _build_hf_headers(self):
            return {"authorization": "token fake"}

    _api = _Api()

    def url(self, path):
        return f"https://example.com/{path}"


def _range_open_from(shard):
    # Stand-in for the remote range read: serve the file-absolute byte range from the local shard.
    def fake_open_range(filesystem, path, file_start, file_end):
        data = shard.read_bytes()[file_start:file_end]
        total = len(shard.read_bytes())
        return _range_cm(
            FakeRangeResponse(206, content_range=f"bytes {file_start}-{file_end - 1}/{total}", chunks=[data])
        )

    return fake_open_range


@pytest.mark.fast
def test_only_the_text_fusion_byte_range_is_extracted_and_then_reused(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    tensors = {
        "transformer_blocks.0.weight": rng.standard_normal((5, 3)),
        "text_fusion.mlp.1.weight": rng.standard_normal((4, 2)),
        "text_fusion.attn.pos": rng.standard_normal((1, 3, 2)),
        "transformer_blocks.1.weight": rng.standard_normal((2,)),
    }
    shard = tmp_path / "shard.safetensors"
    _safetensors(shard, tensors)
    snapshot = tmp_path / "snapshot"
    (snapshot / "transformer").mkdir(parents=True)
    index = {"weight_map": {key: "shard.safetensors" for key in tensors}}
    (snapshot / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
    opened = []

    class LocalFileSystem:
        def open(self, path, mode, block_size=None):
            opened.append(path)
            return shard.open(mode)

    monkeypatch.setattr(zen_image_edit_weights, "HfFileSystem", LocalFileSystem)
    monkeypatch.setattr(zen_image_edit_weights, "_open_range", _range_open_from(shard))
    monkeypatch.setattr(zen_image_edit_weights, "MFLUX_CACHE_DIR", tmp_path / "cache")
    weights = ZenImageEditWeights.text_fusion_weights(snapshot)
    assert sorted(weights) == ["attn.pos", "mlp.1.weight"]
    for key in weights:
        np.testing.assert_array_equal(np.array(weights[key]), tensors[f"text_fusion.{key}"].astype(np.float16))
    assert all(path.endswith("transformer/shard.safetensors") for path in opened)
    assert sorted(p.name for p in (tmp_path / "cache").rglob("*")) == [
        ZenImageEditWeights.revision(),
        "text_fusion.safetensors",
        "zen-image-edit",
    ]
    opened.clear()
    ZenImageEditWeights.text_fusion_weights(snapshot)
    assert opened == []


@pytest.mark.fast
def test_the_zen_repo_is_pinned():
    assert ModelRevisions.get(ZenImageEditWeights.REPO) == "3d5717559ff5bd8b431e66032b285a3fa4a1b20c"


class FakeSmallEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, prompt, images, processor):
        self.calls.append((prompt, len(images)))
        return mx.zeros((1, 3, 8)), mx.array([False, True, False])


@pytest.mark.fast
def test_txt2img_init_skips_the_8b_encoder_and_its_tokenizer(monkeypatch, tmp_path):
    seen = {}
    encoder = FakeSmallEncoder()

    def resolve(path, patterns):
        seen["patterns"] = patterns
        return tmp_path

    def load_components(model, root, definition, quantize, **kwargs):
        seen["definition"] = definition
        model.bits = None

    monkeypatch.setattr(PathResolution, "resolve", resolve)
    monkeypatch.setattr(Qwen21Initializer, "_init_tokenizers", lambda *a: pytest.fail("loaded the 8B tokenizer"))
    monkeypatch.setattr(qwen21_initializer, "Qwen21VAE", lambda: None)
    monkeypatch.setattr(qwen21_initializer, "Qwen21Transformer", lambda: None)
    monkeypatch.setattr(qwen21_initializer.Qwen21SmallTextEncoder, "load", lambda: (encoder, "processor"))
    monkeypatch.setattr(Qwen21Initializer, "load_components", load_components)
    monkeypatch.setattr(Qwen21Initializer, "apply_lora", lambda *a: None)
    model = QwenImage21(small_text_encoder=True)
    assert seen["definition"] is Qwen21NoTextEncoderWeightDefinition
    assert seen["patterns"] == Qwen21NoTextEncoderWeightDefinition.get_download_patterns()
    assert model.text_encoder is encoder and model.processor == "processor" and model.tokenizers == {}

    embeds, mask = model._encode_prompt("a fox")
    model._encode_prompt("a fox")
    assert encoder.calls == [("a fox", 0)]
    assert mask.tolist() == [[1, 1, 1]] and embeds.shape == (1, 3, 8)
    with pytest.raises(ValueError, match="cannot be saved"):
        model.save_model(str(tmp_path / "out"))


@pytest.mark.fast
def test_edit_init_needs_no_text_encoder_folder(monkeypatch, tmp_path):
    for name in ("vae", "transformer"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    encoder = FakeSmallEncoder()
    seen = {}

    def load_components(model, root, definition, quantize, **kwargs):
        seen["definition"] = definition
        model.bits = None

    monkeypatch.setattr(PathResolution, "resolve", lambda *a, **k: tmp_path)
    monkeypatch.setattr(qwen_image21_initializer.TokenizerLoader, "load_all", lambda *a: pytest.fail("8B tokenizer"))
    monkeypatch.setattr(qwen_image21_initializer, "QwenImage21VAE", lambda config: None)
    monkeypatch.setattr(qwen_image21_initializer, "QwenImage21Transformer", lambda config: None)
    monkeypatch.setattr(qwen_image21_initializer.Qwen21SmallTextEncoder, "load", lambda: (encoder, "processor"))
    monkeypatch.setattr(Qwen21Initializer, "load_components", load_components)
    monkeypatch.setattr(Qwen21Initializer, "apply_lora", lambda *a: None)
    model = QwenImage21Edit(small_text_encoder=True)
    assert seen["definition"] is QwenImage21NoTextEncoderWeightDefinition
    assert model.text_encoder is encoder and "text_encoder" not in model._component_configs
    embeds, slots = model._encode_prompt("swap the sky", [])
    assert encoder.calls == [("swap the sky", 0)] and embeds.shape == (1, 3, 8) and slots.shape == (3,)
    with pytest.raises(ValueError, match="native Qwen3-VL"):
        model.rewrite_prompt("make it blue", ["image.png"])
    with pytest.raises(ValueError, match="cannot be saved"):
        model.save_model(str(tmp_path / "out"))


@pytest.mark.fast
@pytest.mark.parametrize("module", [qwen21_generate, qwen21_edit_generate])
def test_both_commands_take_the_flag(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x", "--small-text-encoder"])
    assert module.build_parser().parse_args().small_text_encoder is True
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x"])
    assert module.build_parser().parse_args().small_text_encoder is False


@pytest.mark.fast
@pytest.mark.parametrize("flag", [["--auto-mask", "the shirt"], ["--enhance-prompt"], ["--verify"]])
def test_edit_rejects_the_qwen3_vl_chat_options_with_the_small_encoder(flag, monkeypatch, tmp_path):
    image = tmp_path / "in.png"
    image.write_bytes(b"")
    monkeypatch.setattr(
        sys, "argv", ["prog", "--prompt", "x", "--small-text-encoder", "--image-paths", str(image), *flag]
    )
    parser = qwen21_edit_generate.build_parser()
    args = parser.parse_args()
    with pytest.raises(SystemExit):
        qwen21_edit_generate.validate_edit_args(parser, args, [str(image)])


@pytest.mark.fast
def test_the_vision_tower_loads_in_float32_and_the_language_model_in_bfloat16():
    # In bfloat16 the 12-block vision residual stream drifts to a cosine of ~0.76 on real weights.
    encoder = Qwen21SmallTextEncoder(ENCODER_CONFIG, FUSION_CONFIG)
    encoder.load_encoder_weights(_to_mx(_reference_model().state_dict()))
    dtypes = {
        key: value.dtype for key, value in tree_flatten(encoder.parameters()) if not key.startswith("text_fusion.")
    }
    assert {dtype for key, dtype in dtypes.items() if key.startswith("visual.")} == {mx.float32}
    assert {dtype for key, dtype in dtypes.items() if key.startswith("language_model.")} == {mx.bfloat16}


@pytest.mark.fast
def test_the_vision_tower_runs_in_float32_when_the_language_model_runs_in_bfloat16(monkeypatch):
    encoder = Qwen21SmallTextEncoder(ENCODER_CONFIG, FUSION_CONFIG)
    encoder.load_encoder_weights(_to_mx(_reference_model().state_dict()))
    vision_call = type(encoder.visual).__call__
    seen = {}

    def recording_call(module, pixel_values, grid_thw, return_deepstack=False):
        features, deepstack = vision_call(module, pixel_values, grid_thw, return_deepstack)
        seen["input"], seen["output"] = pixel_values.dtype, features.dtype
        return features, deepstack

    monkeypatch.setattr(type(encoder.visual), "__call__", recording_call)
    ids = mx.array([[1, 2, VISION_START] + [IMAGE_TOKEN] * 6 + [VISION_END, 3]])
    pixels = mx.zeros((24, 3 * 2 * 4 * 4), dtype=mx.bfloat16)
    slices = encoder.hidden_slices(ids, pixels, mx.array([[1, 4, 6]]))
    assert seen == {"input": mx.float32, "output": mx.float32}
    assert slices.dtype == mx.bfloat16


@pytest.mark.fast
def test_a_download_that_ends_early_leaves_no_partial_file(tmp_path, monkeypatch):
    tensors = {"text_fusion.mlp.1.weight": np.ones((64, 64))}
    shard = tmp_path / "shard.safetensors"
    _safetensors(shard, tensors)
    shard.write_bytes(shard.read_bytes()[:-100])  # the remote file stops before the last tensor ends
    snapshot = tmp_path / "snapshot"
    (snapshot / "transformer").mkdir(parents=True)
    index = {"weight_map": {key: "shard.safetensors" for key in tensors}}
    (snapshot / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))

    class LocalFileSystem:
        def open(self, path, mode, block_size=None):
            return shard.open(mode)

    monkeypatch.setattr(zen_image_edit_weights, "HfFileSystem", LocalFileSystem)
    monkeypatch.setattr(zen_image_edit_weights, "_open_range", _range_open_from(shard))
    monkeypatch.setattr(zen_image_edit_weights, "MFLUX_CACHE_DIR", tmp_path / "cache")
    with pytest.raises(RuntimeError, match="ended early"):
        ZenImageEditWeights.text_fusion_weights(snapshot)
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


@pytest.mark.fast
def test_a_range_response_that_is_not_206_is_rejected(monkeypatch):
    # A server that ignores the range request and returns the full body (200) would misalign the bytes.
    monkeypatch.setattr(
        zen_image_edit_weights,
        "http_stream_backoff",
        lambda *a, **k: _range_cm(FakeRangeResponse(200, chunks=[b"full body"])),
    )
    with pytest.raises(OSError, match="Expected HTTP 206"):
        with zen_image_edit_weights._open_range(FakeRangeFileSystem(), "transformer/shard.safetensors", 16, 32):
            pass


@pytest.mark.fast
def test_a_range_response_with_a_mismatched_content_range_is_rejected(monkeypatch):
    monkeypatch.setattr(
        zen_image_edit_weights,
        "http_stream_backoff",
        lambda *a, **k: _range_cm(FakeRangeResponse(206, content_range="bytes 0-15/100", chunks=[b"0123456789abcdef"])),
    )
    with pytest.raises(OSError, match="Content-Range"):
        with zen_image_edit_weights._open_range(FakeRangeFileSystem(), "transformer/shard.safetensors", 16, 32):
            pass


@pytest.mark.fast
def test_a_valid_206_range_response_is_yielded_with_the_requested_range(monkeypatch):
    seen = {}

    def fake_http_stream_backoff(method, url, **kwargs):
        seen["method"], seen["url"], seen["headers"] = method, url, kwargs["headers"]
        return _range_cm(FakeRangeResponse(206, content_range="bytes 16-31/100", chunks=[b"0123456789abcdef"]))

    monkeypatch.setattr(zen_image_edit_weights, "http_stream_backoff", fake_http_stream_backoff)
    with zen_image_edit_weights._open_range(FakeRangeFileSystem(), "transformer/shard.safetensors", 16, 32) as response:
        assert b"".join(response.iter_bytes()) == b"0123456789abcdef"
    assert seen["method"] == "GET"
    assert seen["url"] == "https://example.com/transformer/shard.safetensors"
    assert seen["headers"]["range"] == "bytes=16-31"
    assert seen["headers"]["authorization"] == "token fake"
