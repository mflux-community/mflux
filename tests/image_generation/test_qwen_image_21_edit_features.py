"""Weight-free tests for the three edit features: mask-guided inpainting,
auto-masking through Qwen3-VL grounding, and the first-block step cache."""

import json

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
    def __init__(self, reply=None):
        self.tokenizer = self
        self._reply = reply

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
        return self._reply


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


def _stub_model(tmp_path, reply=None, locate_ids=None):
    model = QwenImage21Edit.__new__(QwenImage21Edit)
    model.model_config = ModelConfig.qwen_image_21()
    model.callbacks = _FakeCallbacks()
    model.tiling_config = None
    model.bits = None
    model.tokenizers = {"qwen21": _FakeTokWrap(reply=reply)}
    model.vae = _FakeVAE()
    model.transformer = None
    model.text_encoder = _FakeTE()
    if locate_ids is not None:
        model.text_encoder.locate_object = lambda *a, **k: locate_ids
    return model


class _RecordingTF:
    transformer_blocks = [0]

    def __init__(self, latent_tokens):
        self.latent_tokens = latent_tokens
        self.calls = []

    def __call_edit__(self, t, config, target_latents, layout, kv_cache=None, kv_cache_mode=None, step_cache=None):
        # MLX arrays are immutable, so holding the reference preserves the step's input
        self.calls.append(target_latents)
        return mx.zeros((1, self.latent_tokens, 64), mx.bfloat16)


# ---------------------------------------------------------------- F1: inpainting


@pytest.mark.fast
def test_latent_mask_block_average() -> None:
    mask = PILImage.new("L", (128, 128), 0)
    mask.paste(PILImage.new("L", (128, 64), 255), (0, 0))
    grid = Qwen21Grounding.to_latent_mask(mask, 8, 8)
    assert grid.shape == (8, 8)
    assert np.allclose(grid[:4], 1.0) and np.allclose(grid[4:], 0.0)


@pytest.mark.fast
def test_inpaint_blend_follows_reference_outside_mask(tmp_path) -> None:
    # With a zero-noise transformer, after each Euler step the unmasked tokens must sit
    # on the reference trajectory x_sigma = blend_source + sigma * (init - blend_source);
    # the stub VAE encodes to zeros, so that reduces to sigma * init there.
    ref = PILImage.new("RGB", (128, 128))
    for x in range(128):
        for y in range(128):
            ref.putpixel((x, y), (x * 2, y * 2, 90))
    mask = PILImage.new("L", (128, 128), 0)  # everything preserved except one corner
    mask.paste(PILImage.new("L", (32, 32), 255), (0, 0))

    tf = _RecordingTF(latent_tokens=8 * 8)
    model = _stub_model(tmp_path)
    model.transformer = tf

    captured = {}

    def fake_to_image(**kwargs):
        captured.update(kwargs)
        return object()

    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    orig_to_image = ImageUtil.to_image
    edit_module.ImageUtil.to_image = staticmethod(fake_to_image)
    try:
        model.generate_image(
            seed=1,
            prompt="p",
            image_paths=[ref],
            num_inference_steps=2,
            width=128,
            height=128,
            mask_image=mask,
            use_kv_cache=False,
        )
    finally:
        edit_module.ImageUtil.to_image = orig_to_image

    step0, step1 = tf.calls
    config = captured["config"]
    sigma1 = float(config.scheduler.sigmas[1])
    init = np.array(step0.astype(mx.float32))
    after = np.array(step1.astype(mx.float32))
    mask_grid = Qwen21Grounding.to_latent_mask(mask, 8, 8).reshape(64, 1)
    unmasked = mask_grid[:, 0] < 0.5
    # stub VAE encodes to zeros -> blend_source is zero -> unmasked = sigma * init
    np.testing.assert_allclose(after[0, unmasked], sigma1 * init[0, unmasked], rtol=2e-2, atol=2e-2)


@pytest.mark.fast
def test_inpaint_pixel_composite_preserves_original(tmp_path) -> None:
    ref = PILImage.new("RGB", (128, 128), (10, 200, 30))
    mask = PILImage.new("L", (128, 128), 0)
    mask.paste(PILImage.new("L", (64, 64), 255), (0, 0))

    model = _stub_model(tmp_path)
    model.transformer = _RecordingTF(latent_tokens=8 * 8)
    captured = {}

    def fake_to_image(**kwargs):
        captured.update(kwargs)
        return object()

    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    orig_to_image = ImageUtil.to_image
    edit_module.ImageUtil.to_image = staticmethod(fake_to_image)
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
        edit_module.ImageUtil.to_image = orig_to_image

    decoded = np.array(captured["decoded_latents"].astype(mx.float32))[0]  # (3, H, W) in [-1, 1]
    original = np.asarray(ref.resize((128, 128)), dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    # stub VAE decodes to zeros -> repainted corner is 0, everything preserved is original
    np.testing.assert_allclose(decoded[:, :64, :64], 0.0, atol=1e-3)
    np.testing.assert_allclose(decoded[:, 64:, :], original[:, 64:, :], atol=1e-4)
    np.testing.assert_allclose(decoded[:, :64, 64:], original[:, :64, 64:], atol=1e-4)


# ----------------------------------------------------------------- F2: grounding


@pytest.mark.fast
def test_grounding_build_input_ids_expands_placeholders() -> None:
    ids = Qwen21Grounding.build_input_ids(_FakeTokWrap(), n_image_tokens=3, query="the red shirt")
    assert ids.count(151655) == 3
    assert "the red shirt" not in str(ids)


@pytest.mark.fast
def test_grounding_parse_bbox_absolute_and_fraction() -> None:
    # absolute pixels of a 200x100 feed
    reply = '```json\n[{"bbox_2d": [10, 20, 110, 80], "label": "shirt"}]\n```'
    assert Qwen21Grounding.parse_bbox(reply, (200, 100)) == (0.05, 0.2, 0.55, 0.8)
    # already-normalized values (<= 2) pass through
    assert Qwen21Grounding.parse_bbox("[[0.1, 0.2, 0.6, 0.8]]", (200, 100)) == (0.1, 0.2, 0.6, 0.8)
    # beyond the shown image's size: the Qwen-VL 0-1000 normalized convention
    assert Qwen21Grounding.parse_bbox("[[359, 168, 580, 783]]", (512, 512)) == (0.359, 0.168, 0.58, 0.783)
    # junk, degenerate, and out-of-range boxes are rejected
    assert Qwen21Grounding.parse_bbox("I cannot see it", (200, 100)) is None
    assert Qwen21Grounding.parse_bbox("[[50, 50, 50, 90]]", (200, 100)) is None
    assert Qwen21Grounding.parse_bbox("[[10, 20, 1100, 1700]]", (200, 100)) is None


@pytest.mark.fast
def test_grounding_rasterize_puts_box_center_white() -> None:
    mask = Qwen21Grounding.rasterize_mask((0.25, 0.25, 0.75, 0.75), (100, 100))
    array = np.asarray(mask)
    assert array[50, 50] == 255
    assert array[2, 2] == 0


@pytest.mark.fast
def test_auto_mask_builds_mask_from_model_reply(tmp_path) -> None:
    ref = PILImage.new("RGB", (128, 128), (10, 200, 30))
    model = _stub_model(tmp_path)
    model.transformer = _RecordingTF(latent_tokens=8 * 8)
    model.text_encoder.locate_object = lambda *a, **k: [7, 8, 9]
    model.tokenizers["qwen21"] = _FakeTokWrap(reply=json.dumps([{"bbox_2d": [32, 32, 96, 96], "label": "x"}]))

    captured = {}

    def fake_to_image(**kwargs):
        captured.update(kwargs)
        return object()

    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    orig_to_image = ImageUtil.to_image
    edit_module.ImageUtil.to_image = staticmethod(fake_to_image)
    try:
        model.generate_image(
            seed=1,
            prompt="p",
            image_paths=[ref],
            num_inference_steps=1,
            width=128,
            height=128,
            auto_mask="the object",
            use_kv_cache=False,
        )
    finally:
        edit_module.ImageUtil.to_image = orig_to_image

    # reply coords are relative to the 512px grounding feed -> fractions 0.0625..0.1875:
    # the repainted spot lands near output pixel (16, 16); the far corner is preserved
    decoded = np.array(captured["decoded_latents"].astype(mx.float32))[0]
    original = np.asarray(ref.resize((128, 128)), dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    np.testing.assert_allclose(decoded[:, 16, 16], 0.0, atol=0.05)  # repainted: stub zeros
    np.testing.assert_allclose(decoded[:, 120, 120], original[:, 120, 120], atol=0.05)


@pytest.mark.fast
def test_auto_mask_failure_raises_actionable_error(tmp_path) -> None:
    ref = PILImage.new("RGB", (128, 128), (10, 200, 30))
    model = _stub_model(tmp_path, reply="not found")
    model.transformer = _RecordingTF(latent_tokens=8 * 8)
    model.text_encoder.locate_object = lambda *a, **k: [7, 8, 9]
    with pytest.raises(ValueError, match="mask_image"):
        model.generate_image(
            seed=1, prompt="p", image_paths=[ref], num_inference_steps=1, auto_mask="nothing", use_kv_cache=False
        )


# ---------------------------------------------------------------- F3: step cache


@pytest.mark.fast
def test_step_cache_signal_accumulation() -> None:
    cache = StepCache(threshold=0.1)
    signal = mx.ones((1, 4, 8))
    assert cache.should_skip(signal) is False  # cold: nothing stored yet
    cache.store(mx.full((1, 4, 8), 2.0), signal)
    assert cache.should_skip(signal) is True  # identical signal -> accum 0 -> skip
    perturbed = signal + mx.full(signal.shape, 0.5)
    # rel diff 0.5 exceeds the threshold: accum resets and a recompute is required
    assert cache.should_skip(perturbed) is False
    cache.store(mx.full((1, 4, 8), 3.0), perturbed)
    assert cache.should_skip(perturbed) is True  # unchanged from the newly stored step


@pytest.mark.fast
def test_step_cache_skips_blocks_on_identical_step(monkeypatch) -> None:
    from mflux.models.common.config.config import Config

    transformer = Qwen21Transformer(num_layers=3)
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
    rng = np.random.default_rng(3)
    text = mx.array(rng.standard_normal((1, 6, 4096)).astype(np.float32)).astype(mx.bfloat16)
    latents = mx.array(rng.standard_normal((1, 16, 64)).astype(np.float32)).astype(mx.bfloat16)

    counter = {"n": 0}
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock

    orig_call = Qwen21TransformerBlock.__call__

    def counting_call(self, *args, **kwargs):
        counter["n"] += 1
        return orig_call(self, *args, **kwargs)

    monkeypatch.setattr(Qwen21TransformerBlock, "__call__", counting_call)

    kv_cache = [None] * 3
    step_cache = StepCache(threshold=0.5)
    layout = [("text", text)]
    out0 = transformer.__call_edit__(
        t=0,
        config=config,
        target_latents=latents,
        layout=layout,
        kv_cache=kv_cache,
        kv_cache_mode="extract",
        step_cache=step_cache,
    )
    assert counter["n"] == 3  # cold step runs every block
    assert step_cache.hidden is not None

    out1 = transformer.__call_edit__(
        t=0,
        config=config,
        target_latents=latents,
        layout=layout,
        kv_cache=kv_cache,
        kv_cache_mode="cached",
        step_cache=step_cache,
    )
    assert counter["n"] == 4  # only block 0 ran; blocks 1-2 reused the cached hidden
    np.testing.assert_allclose(
        np.array(out0.astype(mx.float32)), np.array(out1.astype(mx.float32)), rtol=1e-2, atol=1e-3
    )

    other = mx.array(rng.standard_normal((1, 16, 64)).astype(np.float32)).astype(mx.bfloat16)
    out2 = transformer.__call_edit__(
        t=0,
        config=config,
        target_latents=other,
        layout=layout,
        kv_cache=kv_cache,
        kv_cache_mode="cached",
        step_cache=step_cache,
    )
    assert counter["n"] == 7  # a different input recomputes every block
    assert not np.allclose(np.array(out1.astype(mx.float32)), np.array(out2.astype(mx.float32)), rtol=1e-2, atol=1e-2)


@pytest.mark.fast
def test_step_cache_disabled_matches_baseline(monkeypatch) -> None:
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
    rng = np.random.default_rng(5)
    text = mx.array(rng.standard_normal((1, 6, 4096)).astype(np.float32)).astype(mx.bfloat16)
    latents = mx.array(rng.standard_normal((1, 16, 64)).astype(np.float32)).astype(mx.bfloat16)
    layout = [("text", text)]

    # baseline: prefill then a plain cached step
    kv_baseline = [None] * 2
    transformer.__call_edit__(
        t=0, config=config, target_latents=latents, layout=layout, kv_cache=kv_baseline, kv_cache_mode="extract"
    )
    out_baseline = transformer.__call_edit__(
        t=1, config=config, target_latents=latents, layout=layout, kv_cache=kv_baseline, kv_cache_mode="cached"
    )

    # same with a step cache: an always-skip cache reuses the extract step's hidden,
    # an intentional approximation -- assert the skip happened and the output stays close
    kv_step = [None] * 2
    step_cache = StepCache(threshold=1e9)
    counter = {"n": 0}
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock

    orig_call = Qwen21TransformerBlock.__call__

    def counting_call(self, *args, **kwargs):
        counter["n"] += 1
        return orig_call(self, *args, **kwargs)

    monkeypatch.setattr(Qwen21TransformerBlock, "__call__", counting_call)
    transformer.__call_edit__(
        t=0,
        config=config,
        target_latents=latents,
        layout=layout,
        kv_cache=kv_step,
        kv_cache_mode="extract",
        step_cache=step_cache,
    )
    counter["n"] = 0
    out_cached = transformer.__call_edit__(
        t=1,
        config=config,
        target_latents=latents,
        layout=layout,
        kv_cache=kv_step,
        kv_cache_mode="cached",
        step_cache=step_cache,
    )
    assert counter["n"] == 1  # only block 0 ran
    np.testing.assert_allclose(
        np.array(out_baseline.astype(mx.float32)), np.array(out_cached.astype(mx.float32)), atol=0.25
    )
