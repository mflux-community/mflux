"""Weight-free tests for edit strength, prompt rewriting, and output verification."""

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image as PILImage

from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_grounding import Qwen21Grounding
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit


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
    def __init__(self):
        self.generate_calls = []

    def forward_vl(self, input_ids, pixel_values=None, image_grid_thw=None, image_token_id=151655):
        return mx.zeros((1, int(input_ids.shape[1]), 4096), mx.bfloat16), input_ids == image_token_id

    def locate_object(self, *a, **k):
        return [7, 8, 9]

    def generate(self, input_ids, pixel_values=None, image_grid_thw=None, **k):
        self.generate_calls.append({"n_images": None if image_grid_thw is None else int(image_grid_thw.shape[0])})
        return [101, 102, 103]


def _stub_model(tmp_path, reply=None):
    model = QwenImage21Edit.__new__(QwenImage21Edit)
    model.model_config = ModelConfig.qwen_image_21()
    model.callbacks = _FakeCallbacks()
    model.tiling_config = None
    model.bits = None
    model.tokenizers = {"qwen21": _FakeTokWrap(reply=reply)}
    model.vae = _FakeVAE()
    model.transformer = None
    model.text_encoder = _FakeTE()
    return model


class _RecordingTF:
    transformer_blocks = [0]

    def __init__(self):
        self.inputs = []

    def __call_edit__(self, t, config, target_latents, layout, kv_cache=None, kv_cache_mode=None, step_cache=None):
        self.inputs.append((t, mx.array(np.array(target_latents.astype(mx.float32)))))
        return mx.zeros((1, (config.height // 16) * (config.width // 16), 64), mx.bfloat16)


def _make_ref(tmp_path, size=(128, 128)):
    path = tmp_path / f"ref_{size[0]}x{size[1]}.png"
    PILImage.new("RGB", size, (10, 200, 30)).save(path)
    return str(path)


# ------------------------------------------------------------------ strength


@pytest.mark.fast
def test_strength_starts_midway_with_noised_reference(tmp_path) -> None:
    ref = _make_ref(tmp_path)
    tf = _RecordingTF()
    model = _stub_model(tmp_path)
    model.transformer = tf

    model.generate_image(
        seed=3,
        prompt="p",
        image_paths=[ref],
        num_inference_steps=10,
        width=128,
        height=128,
        strength=0.5,
        use_kv_cache=False,
    )

    assert len(tf.inputs) == 5  # steps after the start index (10 * 0.5)
    first_t, first_latents = tf.inputs[0]
    assert first_t == 5  # sigma-schedule start index, not 0
    # stub VAE encodes to zeros -> blend_source = 0 -> start = sigma_5 * init_noise
    init_noise = np.array(mx.random.normal(key=mx.random.key(3), shape=[1, 64, 64]).astype(mx.float32))
    from mflux.models.common.config.config import Config

    cfg = Config(
        width=128,
        height=128,
        guidance=1.0,
        scheduler="linear",
        image_path=None,
        image_strength=None,
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=10,
    )
    sigma5 = float(cfg.scheduler.sigmas[5])
    # bf16 rounding in the interpolation makes this approximate, not bitwise
    np.testing.assert_allclose(np.array(first_latents)[0], sigma5 * init_noise[0], rtol=0.02, atol=0.02)


@pytest.mark.fast
def test_strength_one_runs_every_step_from_pure_noise(tmp_path) -> None:
    ref = _make_ref(tmp_path)
    tf = _RecordingTF()
    model = _stub_model(tmp_path)
    model.transformer = tf

    model.generate_image(
        seed=3,
        prompt="p",
        image_paths=[ref],
        num_inference_steps=6,
        width=128,
        height=128,
        strength=1.0,
        use_kv_cache=False,
    )
    assert len(tf.inputs) == 6 and tf.inputs[0][0] == 0


@pytest.mark.fast
def test_strength_out_of_range_rejected(tmp_path) -> None:
    ref = _make_ref(tmp_path)
    model = _stub_model(tmp_path)
    model.transformer = _RecordingTF()
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="strength"):
            model.generate_image(
                seed=1,
                prompt="p",
                image_paths=[ref],
                num_inference_steps=2,
                width=128,
                height=128,
                strength=bad,
                use_kv_cache=False,
            )


# ------------------------------------------------------------------ rewrite


@pytest.mark.fast
def test_parse_rewrite_extracts_official_json() -> None:
    reply = 'Sure!\n{"rewritten_prompt": "Change the dress to a flowing red silk evening gown while keeping the pose, face and background unchanged."}\n'
    assert Qwen21Grounding.parse_rewrite(reply).startswith("Change the dress to a flowing red")
    # the official parser's tolerated typo key
    assert Qwen21Grounding.parse_rewrite('{"rewrited_prompt": "x y"}') == "x y"
    # text around multiple braces: last valid span wins
    reply2 = '{"note": "hi"} ... {"rewritten_prompt": "final text"}'
    assert Qwen21Grounding.parse_rewrite(reply2) == "final text"
    assert Qwen21Grounding.parse_rewrite("no json here") is None


@pytest.mark.fast
def test_enhance_prompt_swaps_prompt_and_falls_back(tmp_path) -> None:
    ref = _make_ref(tmp_path)
    model = _stub_model(tmp_path, reply='{"rewritten_prompt": "a much more detailed instruction"}')
    tf = _RecordingTF()
    model.transformer = tf

    captured = {}

    def fake_encode_prompt(self, prompt, pixel_values, grid_thw, ref_latents, ref_shapes, tokenizer, text_encoder):
        captured["prompt"] = prompt
        return [("text", mx.zeros((1, 4, 4096), mx.bfloat16))]

    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    orig = edit_module.QwenImage21Edit._encode_prompt_with_images
    edit_module.QwenImage21Edit._encode_prompt_with_images = fake_encode_prompt
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
    assert captured["prompt"] == "a much more detailed instruction"
    assert len(model.text_encoder.generate_calls) == 1  # one rewrite call, single image

    # unparseable reply -> original prompt used, no crash
    model2 = _stub_model(tmp_path, reply="garbage")
    tf2 = _RecordingTF()
    model2.transformer = tf2
    captured2 = {}
    edit_module.QwenImage21Edit._encode_prompt_with_images = lambda self, prompt, **kw: (
        captured2.__setitem__("prompt", prompt) or [("text", mx.zeros((1, 4, 4096), mx.bfloat16))]
    )
    try:
        model2.generate_image(
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
    assert captured2["prompt"] == "make it red"


# ------------------------------------------------------------------ verification


@pytest.mark.fast
def test_parse_verification_regimes() -> None:
    ok = '{"instruction_applied": true, "outside_unchanged": true}'
    assert Qwen21Grounding.parse_verification(ok) == (True, True)
    half = '{"instruction_applied": true, "outside_unchanged": false}'
    assert Qwen21Grounding.parse_verification(half) == (True, False)
    miss = '{"instruction_applied": false, "outside_unchanged": true}'
    assert Qwen21Grounding.parse_verification(miss) == (False, True)
    assert Qwen21Grounding.parse_verification("the edit looks fine") is None


@pytest.mark.fast
def test_verify_attaches_verdict_and_retries_on_failure(tmp_path) -> None:
    ref = _make_ref(tmp_path)
    replies = [
        '{"instruction_applied": false, "outside_unchanged": false}',  # first: fail
        '{"instruction_applied": true, "outside_unchanged": true}',  # retry: pass
    ]

    class TokWithSequence(_FakeTokWrap):
        def __init__(self):
            super().__init__(reply=replies[0])
            self.i = 0

        def decode(self, ids, **kw):
            r = replies[min(self.i, len(replies) - 1)]
            self.i += 1
            return r

    model = _stub_model(tmp_path)
    model.tokenizers = {"qwen21": TokWithSequence()}
    model.transformer = _RecordingTF()

    class _SimpleImg:
        def __init__(self):
            self.image = PILImage.new("RGB", (128, 128))
            self.verification = None

    # first generate_image call runs the real loop via the original function; the
    # internal retry re-enters through the instance attribute and gets the cheap stub
    # (its verification then consumes the second, passing reply)
    import types

    orig_generate = QwenImage21Edit.generate_image
    inner_calls = {"n": 0}

    def stub_retry(self, **kwargs):
        inner_calls["n"] += 1
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

    assert inner_calls["n"] == 1  # exactly one intercepted retry
    assert result.verification is not None
    assert result.verification["verified"] is True
    assert result.verification["retries"] == 1
