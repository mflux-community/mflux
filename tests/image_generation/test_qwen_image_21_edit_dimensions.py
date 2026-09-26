"""Weight-free regression tests for edit-variant dimension handling and CFG setup:
per-axis explicit dimensions, extreme aspect-ratio validation, generation-time
reporting, and the guidance-without-negative-prompt warning."""

import logging

import mlx.core as mx
import pytest
from PIL import Image as PILImage

from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.utils.image_util import ImageUtil


class _FakeCtx:
    def before_loop(self, latents):
        pass

    def in_loop(self, t, latents, **kw):
        pass

    def after_loop(self, latents):
        pass

    def interruption(self, t, latents):
        pass


class _FakeCallbacks:
    def start(self, **kw):
        return _FakeCtx()


class _FakeTokWrap:
    def __init__(self):
        self.tokenizer = self

    def __call__(self, text, add_special_tokens=False, **kw):
        import re

        ids = []
        parts = text.split("<|image_pad|>")
        for i, chunk in enumerate(parts):
            ids += [1] * len(re.findall(r"\S+", chunk))
            if i < len(parts) - 1:
                ids.append(151655)
        return {"input_ids": [ids]}


class _FakeVAE:
    def encode(self, tensor):
        h, w = int(tensor.shape[2]), int(tensor.shape[3])
        return mx.zeros((1, 64, h // 16, w // 16), mx.bfloat16)

    def decode(self, latents):
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        return mx.zeros((1, 3, int(latents.shape[2]) * 16, int(latents.shape[3]) * 16), mx.bfloat16)


class _FakeTE:
    def forward_vl(self, input_ids, pixel_values=None, image_grid_thw=None, image_token_id=151655):
        return mx.zeros((1, int(input_ids.shape[1]), 4096), mx.bfloat16), input_ids == image_token_id


class _RecordingTF:
    """Stands in for the transformer and records the Config dims of every call."""

    transformer_blocks = [0]

    def __init__(self):
        self.calls = []

    def __call_edit__(self, t, config, target_latents, layout, kv_cache=None, kv_cache_mode=None, step_cache=None):
        self.calls.append((config.width, config.height, t))
        target = (config.height // 16) * (config.width // 16)
        return mx.zeros((1, target, 64), mx.bfloat16)


def _make_ref(tmp_path, size):
    path = tmp_path / f"ref_{size[0]}x{size[1]}.png"
    # asymmetric content so a rotated/wrongly-resized path would also fail these tests
    img = PILImage.new("RGB", size)
    for x in range(0, size[0], max(1, size[0] // 8)):
        for y in range(size[1]):
            img.putpixel((x, y), (x * 255 // max(1, size[0]), y * 255 // max(1, size[1]), 90))
    img.save(path)
    return str(path)


def _stub_model(transformer):
    model = QwenImage21Edit.__new__(QwenImage21Edit)
    model.model_config = ModelConfig.qwen_image_21()
    model.callbacks = _FakeCallbacks()
    model.tiling_config = None
    model.bits = None
    model.tokenizers = {"qwen21": _FakeTokWrap()}
    model.vae = _FakeVAE()
    model.transformer = transformer
    model.text_encoder = _FakeTE()
    return model


@pytest.mark.fast
def test_explicit_axis_is_honored_and_missing_axis_derived(tmp_path) -> None:
    """width=1024 with height=None must produce a 1024-wide image (height derived);
    the pre-fix code overwrote both axes from the aspect ratio."""
    tf = _RecordingTF()
    model = _stub_model(tf)
    ref = _make_ref(tmp_path, (640, 1280))  # 1:2 portrait -> derived pair (736, 1440)

    model.generate_image(
        seed=1, prompt="p", image_paths=[ref], num_inference_steps=1, width=1024, height=None, use_kv_cache=False
    )
    assert tf.calls[-1][:2] == (1024, 1440)  # explicit width kept, height derived

    model.generate_image(
        seed=1, prompt="p", image_paths=[ref], num_inference_steps=1, width=None, height=768, use_kv_cache=False
    )
    assert tf.calls[-1][:2] == (736, 768)  # explicit height kept, width derived

    model.generate_image(
        seed=1, prompt="p", image_paths=[ref], num_inference_steps=1, width=None, height=None, use_kv_cache=False
    )
    assert tf.calls[-1][:2] == (736, 1440)  # both derived from the aspect ratio


@pytest.mark.fast
def test_extreme_aspect_ratio_raises_clear_input_error(tmp_path) -> None:
    """A 300:1 panorama must fail up front with an aspect-ratio message, not a
    ValueError from inside vision preprocessing; a side rounding to 0 (8192:1)
    must fail the same way."""
    model = _stub_model(_RecordingTF())
    for size in [(9600, 32), (8192, 1)]:
        ref = _make_ref(tmp_path, size)
        with pytest.raises(ValueError, match="aspect ratio"):
            model.generate_image(seed=1, prompt="p", image_paths=[ref], num_inference_steps=1, use_kv_cache=False)


@pytest.mark.fast
def test_generation_time_reports_the_iterated_tqdm(monkeypatch, tmp_path) -> None:
    """generation_time must come from the Config-owned tqdm the denoising loop
    iterates; the pre-fix code read a local tqdm that never advanced (always 0)."""
    model = _stub_model(_RecordingTF())
    ref = _make_ref(tmp_path, (512, 512))

    captured = {}

    def fake_to_image(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ImageUtil, "to_image", fake_to_image)

    model.generate_image(seed=1, prompt="p", image_paths=[ref], num_inference_steps=3, use_kv_cache=False)

    config = captured["config"]
    assert config.time_steps.n == 3  # the reported tqdm is the iterated one
    # elapsed keeps ticking after to_image, so compare loosely; the pre-fix bug
    # reported exactly 0.0 from a tqdm that was never iterated
    assert captured["generation_time"] > 0
    assert captured["generation_time"] == pytest.approx(config.time_steps.format_dict["elapsed"], abs=0.1)


@pytest.mark.fast
def test_guidance_without_negative_prompt_warns(monkeypatch, caplog, tmp_path) -> None:
    """guidance > 1 with no negative prompt silently skips CFG; the variant must
    say so (the reference pipeline logs the same warning)."""
    tf = _RecordingTF()
    model = _stub_model(tf)
    ref = _make_ref(tmp_path, (512, 512))

    with caplog.at_level(logging.WARNING, logger="mflux.models.qwen21.variants.edit.qwen_image_21_edit"):
        model.generate_image(
            seed=1,
            prompt="p",
            image_paths=[ref],
            num_inference_steps=2,
            guidance=4.0,
            negative_prompt=None,
            use_kv_cache=False,
        )

    assert any("negative prompt" in r.message for r in caplog.records)
    assert len(tf.calls) == 2  # two steps, one transformer call each: CFG stayed off
