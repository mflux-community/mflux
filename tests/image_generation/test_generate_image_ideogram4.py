import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.model import Ideogram4Config, Ideogram4Transformer, Qwen3TextEncoder
from mflux.models.ideogram4.model.ideogram4_scheduler import Ideogram4Scheduler
from mflux.models.ideogram4.model.ideogram4_text_encoder import (
    Ideogram4Caption,
    Ideogram4CaptionWarning,
    Ideogram4PromptEncoder,
)
from mflux.models.ideogram4.model.ideogram4_transformer import Fp8Linear
from mflux.models.ideogram4.variants import Ideogram4
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition


class FakeTokenizer:
    def tokenize_one(self, prompt: str, max_length=None):  # noqa: ARG002
        return np.asarray([ord(ch) % 31 + 1 for ch in prompt], dtype=np.int64)


class FakeTransformer:
    config = type("Config", (), {"in_channels": 128})()

    def __call__(self, **kwargs):
        return mx.zeros_like(kwargs["x"])


class FakeVAE:
    def decode(self, latents):
        return mx.zeros((latents.shape[0], 3, latents.shape[2] * 8, latents.shape[3] * 8))


class FakeTextEncoder:
    def get_prompt_embeds(self, input_ids, attention_mask, position_ids):  # noqa: ARG002
        return mx.zeros((1, input_ids.shape[1], 53248), dtype=mx.float32)


def _valid_json_caption() -> dict:
    return {
        "high_level_description": "A white ceramic teapot on a simple studio table.",
        "style_description": {
            "aesthetics": "clean, calm, minimal",
            "lighting": "soft diffuse studio lighting",
            "photo": "eye-level, 50mm lens, shallow depth of field",
            "medium": "photograph",
            "color_palette": ["#FFFFFF", "#E5E0D8", "#2E2E2E"],
        },
        "compositional_deconstruction": {
            "background": "A neutral studio tabletop with a pale wall behind it.",
            "elements": [
                {
                    "type": "obj",
                    "bbox": [250, 320, 780, 690],
                    "desc": "A glossy white ceramic teapot with a curved handle and short spout.",
                }
            ],
        },
    }


def _fake_ideogram4_model() -> Ideogram4:
    model = Ideogram4.__new__(Ideogram4)
    model.model_config = ModelConfig.ideogram4_fp8()
    model.callbacks = CallbackRegistry()
    model.tokenizers = {"ideogram4": FakeTokenizer()}
    model.conditional_transformer = FakeTransformer()
    model.unconditional_transformer = FakeTransformer()
    model.vae = FakeVAE()
    model.bits = None
    model.lora_paths = None
    model.lora_scales = None
    model.prompt_cache = {}
    model.text_encoder = FakeTextEncoder()
    return model


def _write_fp8_text_encoder_config(root: Path, *, fp8: bool) -> None:
    config_path = root / "text_encoder" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if fp8:
        config_path.write_text('{"ideogram_fp8_weight_only": true}')
    else:
        config_path.write_text("{}")


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header = {}
    offset = 0
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.extend(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


@pytest.mark.fast
def test_ideogram4_validate_fp8_checkpoint_accepts_fp8_layout(tmp_path: Path) -> None:
    _write_fp8_text_encoder_config(tmp_path, fp8=True)
    assert Ideogram4WeightDefinition.validate_fp8_checkpoint(tmp_path) == tmp_path


@pytest.mark.fast
def test_ideogram4_validate_fp8_checkpoint_skips_mflux_saved_layout(tmp_path: Path) -> None:
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir(parents=True)
    (vae_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"mflux_version": "0.0.0", "quantization_level": "8"}, "weight_map": {}})
    )
    assert Ideogram4WeightDefinition.validate_fp8_checkpoint(tmp_path) == tmp_path


@pytest.mark.fast
def test_ideogram4_validate_fp8_checkpoint_rejects_non_fp8_layout(tmp_path: Path) -> None:
    _write_fp8_text_encoder_config(tmp_path, fp8=False)

    with pytest.raises(ValueError, match="FP8 checkpoint layout"):
        Ideogram4WeightDefinition.validate_fp8_checkpoint(tmp_path)


@pytest.mark.fast
@pytest.mark.parametrize("model_name", ["ideogram4", "ideogram", "ideogram-4-fp8", "ideogram-ai/ideogram-4-fp8"])
def test_ideogram4_alias_detection_accepts_builtin_names(model_name: str) -> None:
    assert ModelConfig.from_name(model_name).model_name == ModelConfig.ideogram4_fp8().model_name


@pytest.mark.fast
def test_ideogram4_alias_detection_rejects_unknown_names() -> None:
    with pytest.raises(Exception):
        ModelConfig.from_name("not-a-real-model-name")


@pytest.mark.fast
def test_ideogram4_generate_cli_treats_alias_as_builtin(monkeypatch, tmp_path: Path) -> None:
    from mflux.models.ideogram4.cli import ideogram4_generate

    captured = {}

    class FakeGeneratedImage:
        def save(self, path: str, export_json_metadata: bool) -> None:
            captured["save_path"] = path
            captured["export_json_metadata"] = export_json_metadata

    class FakeIdeogram4:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def generate_image(self, **kwargs) -> FakeGeneratedImage:
            captured["generate_kwargs"] = kwargs
            return FakeGeneratedImage()

    monkeypatch.setattr(ideogram4_generate, "Ideogram4", FakeIdeogram4)
    monkeypatch.setattr(ideogram4_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-ideogram4",
            "--model",
            "ideogram-4-fp8",
            "--prompt",
            json.dumps(_valid_json_caption()),
            "--seed",
            "1",
            "--width",
            "256",
            "--height",
            "256",
            "--output",
            str(tmp_path / "image.png"),
        ],
    )

    ideogram4_generate.main()

    assert captured["model_path"] is None
    assert captured["model_config"].model_name == "ideogram-ai/ideogram-4-fp8"
    assert captured["save_path"] == str(tmp_path / "image.png")
    assert captured["generate_kwargs"]["preset"] is None
    assert "num_inference_steps" not in captured["generate_kwargs"]


@pytest.mark.fast
def test_ideogram4_generate_cli_warns_when_steps_provided(monkeypatch, tmp_path: Path) -> None:
    from mflux.models.ideogram4.cli import ideogram4_generate

    class FakeGeneratedImage:
        def save(self, path: str, export_json_metadata: bool) -> None:
            pass

    class FakeIdeogram4:
        def __init__(self, **kwargs) -> None:
            pass

        def generate_image(self, **kwargs) -> FakeGeneratedImage:
            return FakeGeneratedImage()

    monkeypatch.setattr(ideogram4_generate, "Ideogram4", FakeIdeogram4)
    monkeypatch.setattr(ideogram4_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-ideogram4",
            "--prompt",
            json.dumps(_valid_json_caption()),
            "--seed",
            "1",
            "--steps",
            "4",
            "--width",
            "256",
            "--height",
            "256",
            "--output",
            str(tmp_path / "image.png"),
        ],
    )

    with pytest.warns(UserWarning, match="--steps is ignored"):
        ideogram4_generate.main()


@pytest.mark.fast
def test_ideogram4_save_cli_treats_alias_as_builtin(monkeypatch, tmp_path: Path) -> None:
    from mflux.models.common.cli import save

    captured = {}

    class FakeIdeogram4:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def save_model(self, path: str) -> None:
            captured["save_path"] = path

    monkeypatch.setattr(save, "Ideogram4", FakeIdeogram4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-save",
            "--model",
            "ideogram-4-fp8",
            "--path",
            str(tmp_path / "saved"),
        ],
    )

    save.main()

    assert captured["model_path"] is None
    assert captured["model_config"].model_name == "ideogram-ai/ideogram-4-fp8"
    assert captured["save_path"] == str(tmp_path / "saved")


@pytest.mark.fast
def test_ideogram4_save_cli_uses_local_path_with_base_model(monkeypatch, tmp_path: Path) -> None:
    from mflux.models.common.cli import save

    captured = {}
    model_path = tmp_path / "checkpoint"

    class FakeIdeogram4:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def save_model(self, path: str) -> None:
            captured["save_path"] = path

    monkeypatch.setattr(save, "Ideogram4", FakeIdeogram4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-save",
            "--model",
            str(model_path),
            "--base-model",
            "ideogram4",
            "--path",
            str(tmp_path / "saved"),
        ],
    )

    save.main()

    assert captured["model_path"] == str(model_path)
    assert captured["model_config"].model_name == "ideogram-ai/ideogram-4-fp8"
    assert captured["save_path"] == str(tmp_path / "saved")


@pytest.mark.fast
@pytest.mark.parametrize("width,height", [(255, 512), (512, 2050), (513, 512)])
def test_ideogram4_validate_dimensions_rejects_bad_sizes(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        Ideogram4LatentCreator.validate_dimensions(width=width, height=height)


@pytest.mark.fast
def test_ideogram4_caption_prepares_dict_as_compact_json() -> None:
    caption = _valid_json_caption()

    prepared = Ideogram4Caption.prepare(caption)

    assert prepared.is_json_caption
    assert prepared.warnings == ()
    assert "\n" not in prepared.prompt
    assert json.loads(prepared.prompt) == caption


@pytest.mark.fast
def test_ideogram4_caption_warns_for_plain_prompt() -> None:
    prepared = Ideogram4Caption.prepare("a white ceramic teapot")

    assert not prepared.is_json_caption
    assert "structured JSON captions" in prepared.warnings[0]


@pytest.mark.fast
def test_ideogram4_caption_warns_for_schema_issues() -> None:
    caption = {
        "style_description": {
            "medium": "photograph",
            "photo": "50mm",
            "lighting": "soft",
            "aesthetics": "clean",
            "color_palette": ["#ffffff"],
        },
        "compositional_deconstruction": {
            "elements": [{"type": "obj", "bbox": [-1, 0, 10, 10], "desc": "A teapot."}],
        },
    }

    messages = Ideogram4Caption.prepare(json.dumps(caption)).warnings

    assert any("background" in message for message in messages)
    assert any("key order" in message for message in messages)
    assert any("uppercase #RRGGBB" in message for message in messages)
    assert any("values should be in [0, 1000]" in message for message in messages)


@pytest.mark.fast
def test_ideogram4_caption_warns_for_invalid_json() -> None:
    prepared = Ideogram4Caption.prepare("{not-json")

    assert not prepared.is_json_caption
    assert "invalid JSON caption" in prepared.warnings[0]
    assert prepared.prompt == "{not-json"


@pytest.mark.fast
def test_ideogram4_fp8_safetensors_reader_handles_raw_dtypes(tmp_path: Path) -> None:
    fp8 = mx.to_fp8(mx.array([[1.0, -2.0], [4.0, -8.0]], dtype=mx.float32))
    mx.eval(fp8)
    scale = np.array([0.5, 0.25], dtype=np.float32)
    bf16_one = np.array([0x3F80], dtype=np.uint16)
    path = tmp_path / "model.safetensors"
    _write_safetensors(
        path,
        {
            "linear.weight": ("F8_E4M3", (2, 2), np.array(fp8).astype(np.uint8).tobytes()),
            "linear.weight_scale": ("F32", (2,), scale.tobytes()),
            "norm.weight": ("BF16", (1,), bf16_one.tobytes()),
        },
    )

    weights = Fp8Linear.read_safetensors(path)

    assert weights["linear.weight"].dtype == mx.uint8
    assert weights["linear.weight"].shape == (2, 2)
    assert weights["linear.weight_scale"].dtype == mx.float32
    assert weights["norm.weight"].dtype == mx.bfloat16
    assert weights["norm.weight"].tolist() == [1.0]


@pytest.mark.fast
def test_ideogram4_fp8_linear_matches_explicit_dequant(tmp_path: Path) -> None:
    fp8 = mx.to_fp8(mx.array([[1.0, -2.0, 3.0], [4.0, -6.0, 8.0]], dtype=mx.float32))
    mx.eval(fp8)
    scale = np.array([0.5, 0.25], dtype=np.float32)
    bias = np.array([0.125, -0.25], dtype=np.float32)
    path = tmp_path / "model.safetensors"
    _write_safetensors(
        path,
        {
            "linear.weight": ("F8_E4M3", (2, 3), np.array(fp8).astype(np.uint8).tobytes()),
            "linear.weight_scale": ("F32", (2,), scale.tobytes()),
            "linear.bias": ("F32", (2,), bias.tobytes()),
        },
    )
    weights = Fp8Linear.read_safetensors(path)
    layer = Fp8Linear(3, 2, bias=True)
    layer.weight = weights["linear.weight"]
    layer.weight_scale = weights["linear.weight_scale"]
    layer.bias = weights["linear.bias"].astype(mx.bfloat16)

    x = mx.array([[[2.0, 3.0, -1.0]]], dtype=mx.bfloat16)
    actual = layer(x)
    dequant = mx.from_fp8(weights["linear.weight"], dtype=mx.bfloat16)
    dequant = dequant * weights["linear.weight_scale"].astype(mx.bfloat16)[:, None]
    expected = mx.matmul(x, mx.transpose(dequant)) + layer.bias
    mx.eval(actual, expected)

    assert np.allclose(np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)), atol=1e-2)


@pytest.mark.fast
def test_ideogram4_scheduler_presets() -> None:
    preset = Ideogram4Scheduler.get_preset("V4_TURBO_12")
    assert preset.num_steps == 12
    assert preset.guidance_schedule == (3.0,) + (7.0,) * 11
    intervals = Ideogram4Scheduler.make_step_intervals(preset.num_steps)
    assert intervals.shape == (13,)
    assert intervals[0] == pytest.approx(0.0)
    assert intervals[-1] == pytest.approx(1.0)


@pytest.mark.fast
def test_ideogram4_qwen_taps_toy_shape() -> None:
    model = Qwen3TextEncoder(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        max_position_embeddings=16,
        head_dim=4,
    )
    ids = mx.array([[1, 2, 3, 0]], dtype=mx.int32)
    mask = mx.array([[1, 1, 1, 0]], dtype=mx.int32)
    pos = mx.array([[0, 1, 2, 0]], dtype=mx.int32)

    embeds = model.get_prompt_embeds(ids, mask, pos, tap_layers=(0, 1))
    mx.eval(embeds)

    assert embeds.shape == (1, 4, 16)


@pytest.mark.fast
def test_ideogram4_transformer_toy_shape() -> None:
    config = Ideogram4Config(
        emb_dim=12,
        num_layers=1,
        num_heads=3,
        intermediate_size=16,
        adanln_dim=4,
        in_channels=4,
        llm_features_dim=8,
        mrope_section=(1, 0, 0),
    )
    model = Ideogram4Transformer(config)
    output = model(
        llm_features=mx.zeros((1, 5, 8), dtype=mx.float32),
        x=mx.zeros((1, 5, 4), dtype=mx.float32),
        t=mx.array([0.5], dtype=mx.float32),
        position_ids=mx.zeros((1, 5, 3), dtype=mx.int32),
        segment_ids=mx.ones((1, 5), dtype=mx.int32),
        indicator=mx.array([[3, 3, 2, 2, 2]], dtype=mx.int32),
    )
    mx.eval(output)

    assert output.shape == (1, 5, 4)
    assert output.dtype == mx.float32


@pytest.mark.fast
def test_ideogram4_tokenizer_max_length_matches_official_limit() -> None:
    model_config = ModelConfig.ideogram4_fp8()
    tokenizer_defs = Ideogram4WeightDefinition.get_tokenizers()
    assert len(tokenizer_defs) == 1
    assert tokenizer_defs[0].max_length == 2048
    assert model_config.max_sequence_length == tokenizer_defs[0].max_length


@pytest.mark.fast
def test_ideogram4_build_inputs_packs_text_and_image_tokens() -> None:
    model = Ideogram4.__new__(Ideogram4)
    model.tokenizers = {"ideogram4": FakeTokenizer()}

    inputs = Ideogram4PromptEncoder.build_inputs(
        model.tokenizers["ideogram4"],
        ["abc"],
        height=256,
        width=256,
    )

    assert inputs["num_image_tokens"] == 256
    assert inputs["max_text_tokens"] == 3
    indicator = inputs["indicator"].tolist()[0]
    assert indicator[:3] == [3] * 3
    assert indicator[3:] == [2] * 256
    assert inputs["position_ids"].tolist()[0][3] == [65536] * 3


@pytest.mark.fast
def test_ideogram4_generate_image_with_fake_components() -> None:
    model = _fake_ideogram4_model()
    shift, scale = Ideogram4LatentCreator.get_latent_norm()
    assert shift.shape == (128,)
    assert scale.shape == (128,)

    image = model.generate_image(prompt=_valid_json_caption(), seed=1, num_inference_steps=1, width=256, height=256)

    assert image.image.size == (256, 256)
    assert Ideogram4LatentCreator.unpack_latents(mx.zeros((1, 256, 128)), 256, 256).shape == (1, 32, 32, 32)


@pytest.mark.fast
def test_ideogram4_generate_image_rejects_non_positive_steps() -> None:
    model = _fake_ideogram4_model()

    with pytest.raises(ValueError, match="num_inference_steps must be >= 1"):
        model.generate_image(prompt=_valid_json_caption(), seed=1, num_inference_steps=0, width=256, height=256)


@pytest.mark.fast
def test_ideogram4_generate_image_warns_for_plain_prompt() -> None:
    model = _fake_ideogram4_model()

    with pytest.warns(Ideogram4CaptionWarning, match="plain prompt"):
        model.generate_image(prompt="abc", seed=1, num_inference_steps=1, width=256, height=256)


@pytest.mark.fast
def test_ideogram4_generate_image_strict_caption_validation_fails_on_plain_prompt() -> None:
    model = _fake_ideogram4_model()

    with pytest.raises(ValueError, match="plain prompt"):
        model.generate_image(
            prompt="abc",
            seed=1,
            num_inference_steps=1,
            width=256,
            height=256,
            strict_caption_validation=True,
        )
