"""Integration tests for the edit capabilities: feature combinations, retry
exhaustion, fallback paths, KV-cache equivalence at a shifted strength start,
and the CLI flag-to-API contract. Heavy models are stubbed; everything else
(plumbing, blending, scheduling, retries) runs for real."""

import types

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image as PILImage

from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_grounding import Qwen21Grounding
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer, StepCache
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
    def __init__(self, replies=None):
        self.tokenizer = self
        self.replies = list(replies or [])

    def __call__(self, text, add_special_tokens=False, **kw):
        import re

        ids = []
        parts = text.split("<|image_pad|>")
        for i, chunk in enumerate(parts):
            ids += [1] * len(re.findall(r"\S+", chunk))
            if i < len(parts) - 1:
                ids.append(151655)
        return {"input_ids": [ids]}

    def decode(self, ids, **kw):
        if not self.replies:
            return "ok"
        return self.replies.pop(0) if len(self.replies) == 1 else self.replies.pop(0)


class _StubVAE:
    """decode honors keep_alpha like the real VAE: 4 channels when requested."""

    def __init__(self):
        self.encode_inputs = []

    def encode(self, tensor):
        self.encode_inputs.append(mx.array(np.array(tensor.astype(mx.float32))))
        h, w = int(tensor.shape[2]), int(tensor.shape[3])
        return mx.zeros((1, 64, h // 16, w // 16), mx.bfloat16)

    def decode(self, latents, keep_alpha=False):
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        h, w = int(latents.shape[2]) * 16, int(latents.shape[3]) * 16
        channels = 4 if keep_alpha else 3
        return mx.zeros((1, channels, h, w), mx.bfloat16)


class _FakeTE:
    def __init__(self, fail_generate=False):
        self.fail_generate = fail_generate

    def forward_vl(self, input_ids, pixel_values=None, image_grid_thw=None, image_token_id=151655):
        return mx.zeros((1, int(input_ids.shape[1]), 4096), mx.bfloat16), input_ids == image_token_id

    def locate_object(self, *a, **k):
        return [7, 8, 9]

    def generate(self, input_ids, pixel_values=None, image_grid_thw=None, **k):
        if self.fail_generate:
            raise RuntimeError("simulated encoder failure")
        return [101, 102]


class _RecordingTF:
    transformer_blocks = [0]

    def __init__(self):
        self.inputs = []

    def __call_edit__(self, t, config, target_latents, layout, kv_cache=None, kv_cache_mode=None, step_cache=None):
        self.inputs.append((t, np.array(target_latents.astype(mx.float32))))
        return mx.zeros((1, (config.height // 16) * (config.width // 16), 64), mx.bfloat16)


def _stub_model(tmp_path, replies=None, fail_generate=False):
    model = QwenImage21Edit.__new__(QwenImage21Edit)
    model.model_config = ModelConfig.qwen_image_21()
    model.callbacks = _FakeCallbacks()
    model.tiling_config = None
    model.bits = None
    model.tokenizers = {"qwen21": _FakeTokWrap(replies)}
    model.vae = _StubVAE()
    model.transformer = _RecordingTF()
    model.text_encoder = _FakeTE(fail_generate)
    return model


def _ref(tmp_path, size=(128, 128), color=(10, 200, 30)):
    path = tmp_path / f"ref_{size[0]}x{size[1]}.png"
    PILImage.new("RGB", size, color).save(path)
    return str(path)


def _corner_mask(tmp_path, size=(128, 128), quadrant=(32, 32)):
    mask = PILImage.new("L", size, 0)
    mask.paste(PILImage.new("L", quadrant, 255), (0, 0))
    path = tmp_path / f"mask_{size[0]}.png"
    mask.save(path)
    return str(path)


def _capture_to_image(monkeypatch_module):
    captured = {}

    class _Wrapped:
        # mirrors GeneratedImage: .image is the PIL image
        def __init__(self, img):
            self.image = img
            self.verification = None

    def fake_to_image(**kwargs):
        captured.update(kwargs)
        img = ImageUtil.to_pil(kwargs["decoded_latents"])
        captured["pil"] = img
        captured["pil_mode"] = img.mode
        return _Wrapped(img)

    orig = monkeypatch_module.ImageUtil.to_image
    monkeypatch_module.ImageUtil.to_image = staticmethod(fake_to_image)
    return captured, orig


# ------------------------------------------------------- CLI flag surface


@pytest.mark.fast
def test_cli_forwards_full_flag_surface(monkeypatch, tmp_path) -> None:
    import sys

    from mflux.models.qwen21.cli import qwen21_edit_generate as cli

    src = _ref(tmp_path)
    mask = _corner_mask(tmp_path)
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            self.callbacks = types.SimpleNamespace(register=lambda *a, **kw: None)

        def generate_image(self, **kwargs):
            captured.update(kwargs)
            img = types.SimpleNamespace(save=lambda **kw: None, verification=None)
            return img

    monkeypatch.setattr(cli, "QwenImage21Edit", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-2.1-edit",
            "--image-paths",
            src,
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "out_{seed}.png"),
            "--mask-image",
            mask,
            "--strength",
            "0.7",
            "--use-step-cache",
            "--step-cache-threshold",
            "0.2",
            "--enhance-prompt",
            "--verify",
            "--verify-retries",
            "1",
            "--rgba-output",
            "--steps",
            "2",
        ],
    )
    cli.main()

    expected = {
        "mask_image": mask,
        "strength": 0.7,
        "use_step_cache": True,
        "step_cache_threshold": 0.2,
        "enhance_prompt": True,
        "verify": True,
        "verify_retries": 1,
        "rgba_output": True,
        "num_inference_steps": 2,
    }
    for key, value in expected.items():
        assert captured[key] == value, f"{key}: {captured.get(key)!r} != {value!r}"


# ------------------------------------------------------- strength interactions


@pytest.mark.fast
def test_strength_one_is_bitwise_default_run(tmp_path) -> None:
    ref = _ref(tmp_path)
    model = _stub_model(tmp_path)
    model.generate_image(
        seed=9, prompt="p", image_paths=[ref], num_inference_steps=3, width=128, height=128, use_kv_cache=False
    )
    first_run = list(model.transformer.inputs)
    model.transformer = _RecordingTF()
    model.generate_image(
        seed=9,
        prompt="p",
        image_paths=[ref],
        num_inference_steps=3,
        width=128,
        height=128,
        strength=1.0,
        use_kv_cache=False,
    )
    assert len(first_run) == len(model.transformer.inputs) == 3
    for (t1, lat1), (t2, lat2) in zip(first_run, model.transformer.inputs):
        assert t1 == t2
        np.testing.assert_array_equal(lat1, lat2)


@pytest.mark.fast
def test_kv_cache_matches_uncached_at_shifted_strength_start() -> None:
    from mflux.models.common.config.config import Config

    transformer = Qwen21Transformer(num_layers=2)
    config = Config(
        width=64,
        height=64,
        guidance=1.0,
        scheduler="linear",
        image_path=None,
        image_strength=None,
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=4,
    )
    rng = np.random.default_rng(4)
    text = mx.array(rng.standard_normal((1, 6, 4096)).astype(np.float32)).astype(mx.bfloat16)
    latents = mx.array(rng.standard_normal((1, 16, 64)).astype(np.float32)).astype(mx.bfloat16)
    layout = [("text", text)]

    kv = [None] * 2
    transformer.__call_edit__(
        t=2, config=config, target_latents=latents, layout=layout, kv_cache=kv, kv_cache_mode="extract"
    )
    out_cached = transformer.__call_edit__(
        t=3, config=config, target_latents=latents, layout=layout, kv_cache=kv, kv_cache_mode="cached"
    )
    out_plain = transformer.__call_edit__(t=3, config=config, target_latents=latents, layout=layout)
    np.testing.assert_allclose(
        np.array(out_cached.astype(mx.float32)), np.array(out_plain.astype(mx.float32)), atol=1e-2
    )


@pytest.mark.fast
def test_mask_and_strength_compose(tmp_path) -> None:
    ref = _ref(tmp_path)
    mask = _corner_mask(tmp_path)
    model = _stub_model(tmp_path)
    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    captured, orig = _capture_to_image(edit_module)
    try:
        model.generate_image(
            seed=1,
            prompt="p",
            image_paths=[ref],
            num_inference_steps=10,
            width=128,
            height=128,
            mask_image=mask,
            strength=0.5,
            use_kv_cache=False,
            rgba_output=True,
        )
    finally:
        edit_module.ImageUtil.to_image = orig
    # 5 denoising steps after the strength start, RGBA composite executed
    assert len(model.transformer.inputs) == 5
    assert model.transformer.inputs[0][0] == 5  # schedule index, not 0
    assert captured["decoded_latents"].shape[1] == 4
    assert captured["pil_mode"] == "RGBA"
    # the composite split correctly: the masked corner carries the stub-decoded
    # zeros, the preserved region carries the original pixels (alpha padded to 1)
    decoded = np.array(captured["decoded_latents"].astype(mx.float32))[0]
    np.testing.assert_allclose(decoded[:, 5, 5], 0.0, atol=1e-4)
    original = np.asarray(PILImage.new("RGB", (128, 128), (10, 200, 30)), dtype=np.float32) / 127.5 - 1.0
    np.testing.assert_allclose(decoded[:3, 100, 100], original[100, 100], atol=1e-4)
    np.testing.assert_allclose(decoded[3, 100, 100], 1.0, atol=1e-4)


# ------------------------------------------------------- verification paths


@pytest.mark.fast
def test_verify_retry_exhaustion_returns_original_with_failed_verdict(tmp_path) -> None:
    ref = _ref(tmp_path)
    fail_reply = '{"instruction_applied": false, "outside_unchanged": false}'
    model = _stub_model(tmp_path, replies=[fail_reply, fail_reply])  # both verdicts fail

    class _SimpleImg:
        def __init__(self):
            self.image = PILImage.new("RGB", (128, 128))
            self.verification = None

    orig_generate = QwenImage21Edit.generate_image
    inner = {"n": 0}

    def stub_retry(self, **kwargs):
        inner["n"] += 1
        return _SimpleImg()

    model.generate_image = types.MethodType(stub_retry, model)
    try:
        result = orig_generate(
            model,
            seed=5,
            prompt="p",
            image_paths=[ref],
            num_inference_steps=1,
            width=128,
            height=128,
            verify=True,
            verify_retries=1,
            use_kv_cache=False,
        )
    finally:
        del model.generate_image
    assert inner["n"] == 1  # one retry was attempted
    assert result.verification["verified"] is False
    assert result.verification["retries"] == 1


@pytest.mark.fast
def test_verify_unparseable_reply_reports_parse_failure(tmp_path) -> None:
    ref = _ref(tmp_path)
    model = _stub_model(tmp_path, replies=["I cannot tell"])
    result = model.generate_image(
        seed=5,
        prompt="p",
        image_paths=[ref],
        num_inference_steps=1,
        width=128,
        height=128,
        verify=True,
        use_kv_cache=False,
    )
    assert result.verification["parse_failed"] is True
    assert result.verification["verified"] is False


@pytest.mark.fast
def test_enhance_prompt_falls_back_when_encoder_raises(tmp_path) -> None:
    ref = _ref(tmp_path)
    model = _stub_model(tmp_path, fail_generate=True)  # generate() raises
    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    captured = {}

    def fake_encode(self, prompt, **kwargs):
        captured["prompt"] = prompt
        return [("text", mx.zeros((1, 4, 4096), mx.bfloat16))]

    orig = edit_module.QwenImage21Edit._encode_prompt_with_images
    edit_module.QwenImage21Edit._encode_prompt_with_images = fake_encode
    try:
        model.generate_image(
            seed=1,
            prompt="make it red",
            image_paths=[ref],
            num_inference_steps=1,
            width=128,
            height=128,
            enhance_prompt=True,
            use_kv_cache=False,
        )
    finally:
        edit_module.QwenImage21Edit._encode_prompt_with_images = orig
    assert captured["prompt"] == "make it red"  # original used, no crash


# ------------------------------------------------------- tokenization / cache units


@pytest.mark.fast
def test_tokenize_with_images_expands_placeholders_independently() -> None:
    text = "A<|vision_start|><|image_pad|><|vision_end|> B<|vision_start|><|image_pad|><|vision_end|>"
    ids = Qwen21Grounding.tokenize_with_images(_FakeTokWrap(), text, [2, 3])
    assert ids.count(151655) == 5  # 2 + 3, in order: first two, then three


@pytest.mark.fast
def test_step_cache_accumulates_until_threshold() -> None:
    cache = StepCache(threshold=0.1)
    base = mx.ones((1, 4, 8))
    cache.store(base, base)
    small1 = base + 0.04
    small2 = base + 0.08
    assert cache.should_skip(small1) is True  # accum 0.04 < 0.1 -> skip
    assert cache.should_skip(small2) is False  # accum 0.04 + 0.08 = 0.12 >= 0.1 -> recompute, reset
    assert cache.should_skip(small1) is True  # fresh accumulator after the reset


@pytest.mark.fast
def test_mask_smaller_than_output_is_resized(tmp_path) -> None:
    ref = _ref(tmp_path)
    mask = _corner_mask(tmp_path, size=(64, 64))  # half the output size
    model = _stub_model(tmp_path)
    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    captured, orig = _capture_to_image(edit_module)
    try:
        model.generate_image(
            seed=1,
            prompt="p",
            image_paths=[ref],
            num_inference_steps=1,
            width=128,
            height=128,
            mask_image=mask,
            use_kv_cache=False,
        )
    finally:
        edit_module.ImageUtil.to_image = orig
    # decoded composite ran at output size without error
    assert captured["decoded_latents"].shape[-2:] == (128, 128)


# ------------------------------------------------------- full-pipeline smoke


@pytest.mark.fast
def test_all_features_combined_smoke(tmp_path) -> None:
    """enhance + mask + strength + step cache + verify + rgba in one call."""
    ref = _ref(tmp_path)
    mask = _corner_mask(tmp_path)
    model = _stub_model(
        tmp_path,
        replies=[
            '{"rewritten_prompt": "a detailed description"}',  # enhance
            '{"instruction_applied": true, "outside_unchanged": true}',  # verify
        ],
    )
    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    captured, orig = _capture_to_image(edit_module)
    try:
        result = model.generate_image(
            seed=11,
            prompt="p",
            image_paths=[ref],
            num_inference_steps=4,
            width=128,
            height=128,
            mask_image=mask,
            strength=0.75,
            use_step_cache=True,
            step_cache_threshold=0.9,
            enhance_prompt=True,
            verify=True,
            rgba_output=True,
            use_kv_cache=False,
        )
    finally:
        edit_module.ImageUtil.to_image = orig
    # 3 steps after the strength start (int(4 * 0.75) == 3 -> steps t=3 only)
    assert len(model.transformer.inputs) == 1
    assert captured["pil_mode"] == "RGBA"
    assert result.verification["verified"] is True
    assert result.verification["retries"] == 0
