import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from mflux.callbacks.callback_manager import CallbackManager
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.cli import (
    qwen21_edit_generate as cli,
    qwen21_generate,
)
from mflux.models.qwen21.model.qwen21_text_encoder.grounding import QwenImage21Grounding
from mflux.models.qwen21.model.qwen21_text_encoder.text_encoder import QwenImage21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import (
    QwenImage21Transformer,
    StepCache,
)
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

pytestmark = pytest.mark.fast

SOURCE_RGB = (200, 30, 10)


class _FakeTransformer:
    axes = (4, 6, 6)

    def __init__(self):
        self.calls = []

    def __call__(self, hidden, text, timestep, layout, cache=None, step_cache=None):
        self.calls.append(float(timestep[0]))
        return mx.ones((1, layout.target_tokens, hidden.shape[-1]))


class _FakeVAE:
    def __init__(self):
        self.encoded = []

    def encode(self, pixels):
        h, w = pixels.shape[2] // 16, pixels.shape[3] // 16
        self.encoded.append(float(mx.mean(pixels)))
        return mx.full((1, 64, 1, h, w), self.encoded[-1])

    def decode(self, latents):
        # zeros: repainted pixels come back mid-gray, preserved ones must be the source
        return mx.zeros((1, 4, 1, latents.shape[3] * 16, latents.shape[4] * 16))


class _Recorder:
    def __init__(self):
        self.seeds = []
        self.latents = None

    def call_before_loop(self, seed, prompt, latents, config, **kwargs):
        self.seeds.append(seed)

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        self.latents = latents


def _stub_model(prompts=None):
    model = QwenImage21Edit.__new__(QwenImage21Edit)
    model.model_config = ModelConfig.qwen_image_21()
    model.processor = SimpleNamespace(image_processor=SimpleNamespace(size={"shortest_edge": 0, "longest_edge": 1e12}))
    model.prompt_cache = {}
    model.text_encoder = SimpleNamespace()
    model.transformer = _FakeTransformer()
    model.vae = _FakeVAE()
    model.callbacks = CallbackRegistry()
    model.tiling_config = None
    model.bits = None
    model.lora_paths = None
    model.lora_scales = None
    model.compute_precision = ComputePrecision()

    def encode(prompt, images):
        if prompts is not None:
            prompts.append(prompt)
        # 64x64 reference at output_resolution 64 -> 4x4 latents -> 4 slots of 4 tokens
        return mx.zeros((1, 6, 8)), mx.array([False, True, True, True, True, False])

    model._encode_prompt = encode
    return model


def _source(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGBA", (64, 64), (*SOURCE_RGB, 255)).save(path)
    return str(path)


def _generate(model, tmp_path, seed=1, **kwargs):
    return model.generate_image(
        seed=seed,
        prompt="edit",
        num_inference_steps=4,
        image_paths=[_source(tmp_path)],
        output_resolution=64,
        **kwargs,
    )


def _left_half_mask(tmp_path):
    path = tmp_path / "mask.png"
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (0, 0, 32, 64))
    mask.save(path)
    return str(path)


def test_mask_image_repaints_only_the_white_region(tmp_path):
    image = _generate(_stub_model(), tmp_path, mask_image=_left_half_mask(tmp_path))
    pixels = np.asarray(image.image.convert("RGB")).astype(int)
    np.testing.assert_allclose(pixels[32, 8], (128, 128, 128), atol=2)  # repainted (decoder zeros)
    np.testing.assert_allclose(pixels[32, 56], SOURCE_RGB, atol=2)  # preserved exactly


def test_unmasked_latents_end_on_the_source(tmp_path):
    # The final pixel composite hides the per-step blend, so check the latents directly.
    model = _stub_model()
    recorder = _Recorder()
    model.callbacks.register(recorder)
    _generate(model, tmp_path, mask_image=_left_half_mask(tmp_path))
    # 64x64 output -> 4x4 latent grid. The model repaints the white left half (columns 0-1).
    tokens = np.array(recorder.latents).reshape(4, 4, -1)
    source = model.vae.encoded[-1]
    # sigma is 0 after the final step, so unmasked tokens equal the encoded source exactly
    np.testing.assert_array_equal(tokens[:, 2:], np.full_like(tokens[:, 2:], source))
    assert not np.allclose(tokens[:, :2], source)


def test_auto_mask_reads_qwen3_vl_0_1000_boxes(tmp_path):
    model = _stub_model()
    model._vision_reply = lambda instruction, images, tokens: '[{"bbox_2d": [0, 0, 500, 1000], "label": "x"}]'
    pixels = np.asarray(_generate(model, tmp_path, auto_mask="the left half").image.convert("RGB")).astype(int)
    np.testing.assert_allclose(pixels[32, 8], (128, 128, 128), atol=2)
    np.testing.assert_allclose(pixels[32, 56], SOURCE_RGB, atol=2)


def test_auto_mask_without_a_box_is_an_actionable_error(tmp_path):
    model = _stub_model()
    model._vision_reply = lambda instruction, images, tokens: "I cannot see it"
    with pytest.raises(ValueError, match="provide mask_image"):
        _generate(model, tmp_path, auto_mask="a unicorn")


@pytest.mark.parametrize("strength,calls", [(1.0, 4), (0.5, 2), (0.25, 1)])
def test_strength_skips_the_start_of_the_schedule(tmp_path, strength, calls):
    model = _stub_model()
    _generate(model, tmp_path, strength=strength)
    assert len(model.transformer.calls) == calls


def test_enhance_prompt_encodes_the_rewrite_and_records_the_original(tmp_path):
    prompts = []
    model = _stub_model(prompts)
    model._vision_reply = lambda instruction, images, tokens: '{"rewritten_prompt": "a detailed edit"}'
    image = _generate(model, tmp_path, enhance_prompt=True)
    assert prompts == ["a detailed edit"]
    assert image.generation_parameters["original_prompt"] == "edit"


def test_enhance_prompt_shows_the_rewrite_every_reference_image(tmp_path):
    # Only the first image used to reach the rewrite, so an instruction about <image2> came back
    # as "<image2> is not present" (#831).
    model = _stub_model()
    seen = []

    def reply(instruction, images, tokens):
        seen.append((instruction, len(images)))
        return '{"rewritten_prompt": "a detailed edit"}'

    model._vision_reply = reply

    class _Encoded(Exception):
        pass

    def stop(prompt, images):
        raise _Encoded  # the stubbed encoder serves one reference; the rewrite is what this checks

    model._encode_prompt = stop
    second = tmp_path / "second.png"
    Image.new("RGBA", (64, 64), (10, 200, 30, 255)).save(second)
    with pytest.raises(_Encoded):
        model.generate_image(
            seed=1,
            prompt="put the hat from <image2> on <image1>",
            num_inference_steps=4,
            image_paths=[_source(tmp_path), str(second)],
            output_resolution=64,
            enhance_prompt=True,
        )
    instruction, count = seen[0]
    assert count == 2
    assert "<image1>, the image to be edited, and the reference image <image2>" in instruction


def test_rewrite_prompt_is_public_and_matches_enhance_prompt(tmp_path):
    prompts = []
    model = _stub_model(prompts)
    model._vision_reply = lambda instruction, images, tokens: '{"rewritten_prompt": "a detailed edit"}'

    rewritten = model.rewrite_prompt("edit", [_source(tmp_path)])
    model.generate_image(
        seed=1, prompt=rewritten, num_inference_steps=4, image_paths=[_source(tmp_path)], output_resolution=64
    )

    assert rewritten == "a detailed edit"
    assert prompts == ["a detailed edit"]


def test_rewrite_prompt_raises_when_it_cannot_rewrite(tmp_path):
    # enhance_prompt falls back to the instruction; the public method says so instead, or the caller
    # would generate from the terse instruction thinking it was rewritten.
    model = _stub_model()
    model._vision_reply = lambda instruction, images, tokens: "no json here"
    with pytest.raises(ValueError, match="could not be parsed"):
        model.rewrite_prompt("edit", [_source(tmp_path)])

    released = _stub_model()
    released.text_encoder = None
    with pytest.raises(RuntimeError, match="released by the memory saver"):
        released.rewrite_prompt("edit", [_source(tmp_path)])


def test_enhance_prompt_falls_back_when_the_text_encoder_was_released(tmp_path):
    prompts = []
    model = _stub_model(prompts)
    model.text_encoder = None
    _generate(model, tmp_path, enhance_prompt=True)
    assert prompts == ["edit"]


def test_scheduler_reaches_the_edit_schedule(tmp_path):
    # The CLI checked --scheduler viggle_turbo and then never passed it on (#831).
    model = _stub_model()
    schedulers = []

    class _SchedulerRecorder(_Recorder):
        def call_before_loop(self, seed, prompt, latents, config, **kwargs):
            schedulers.append(type(config.scheduler).__name__)

    model.callbacks.register(_SchedulerRecorder())
    model.generate_image(
        seed=1,
        prompt="edit",
        num_inference_steps=6,
        image_paths=[_source(tmp_path)],
        output_resolution=64,
        scheduler="viggle_turbo",
    )

    assert schedulers == ["ViggleTurboScheduler"]


def test_cli_passes_the_scheduler_to_the_edit(tmp_path, monkeypatch):
    calls = []

    class _Model:
        def __init__(self, **kwargs):
            self.callbacks = CallbackRegistry()

        def generate_image(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(save=lambda *args, **kwargs: None, verification=None)

    monkeypatch.setattr(cli, "QwenImage21Edit", _Model)
    monkeypatch.setattr(cli.CallbackManager, "register_callbacks", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mflux-generate-qwen-2.1-edit", "--prompt", "edit", "--image-paths", _source(tmp_path), "--steps", "6",
         "--scheduler", "viggle_turbo", "--output", str(tmp_path / "out.png")],
    )  # fmt: skip

    cli.main()

    assert calls and calls[0]["scheduler"] == "viggle_turbo"


@pytest.mark.parametrize("scheduler", ["linear", "viggle_turbo"])
def test_edit_records_a_scheduler_other_than_linear(tmp_path, scheduler):
    # The schedule changes the image, so --config-from-conf has to find it in the sidecar.
    image = _stub_model().generate_image(
        seed=1,
        prompt="edit",
        num_inference_steps=6,
        image_paths=[_source(tmp_path)],
        output_resolution=64,
        scheduler=scheduler,
    )

    assert image.generation_parameters.get("scheduler") == (None if scheduler == "linear" else scheduler)


@pytest.mark.parametrize("command", [cli, qwen21_generate])
def test_config_from_conf_replays_the_scheduler(tmp_path, monkeypatch, command):
    sidecar = tmp_path / "image.metadata.json"
    recorded = {
        "prompt": "edit",
        "seed": 1,
        "steps": 6,
        "scheduler": "viggle_turbo",
        "image_paths": [_source(tmp_path)],
    }
    sidecar.write_text(json.dumps(recorded))

    monkeypatch.setattr(sys, "argv", ["mflux", "--config-from-conf", str(sidecar)])
    assert command.build_parser().parse_args().scheduler == "viggle_turbo"
    monkeypatch.setattr(sys, "argv", ["mflux", "--config-from-conf", str(sidecar), "--scheduler", "linear"])
    assert command.build_parser().parse_args().scheduler == "linear"


def test_rewrite_prompt_takes_at_most_ten_references():
    # generate_image refuses an eleventh reference, so a rewrite that names <image11> could not be used.
    with pytest.raises(ValueError, match="at most 10"):
        _stub_model().rewrite_prompt("edit", ["unread.png"] * 11)


def test_image_paths_take_in_memory_images(tmp_path):
    # Images passed directly work like paths, also with strength < 1, and the metadata records a
    # placeholder instead of the object's repr, which carries a memory address (#831).
    model = _stub_model()
    source = Image.new("RGBA", (64, 64), (*SOURCE_RGB, 255))

    image = model.generate_image(
        seed=1, prompt="edit", num_inference_steps=4, image_paths=[source], output_resolution=64, strength=0.5
    )

    assert image.image_paths == ["<in-memory image>"]


def test_enhance_prompt_falls_back_on_an_unparseable_reply(tmp_path):
    prompts = []
    model = _stub_model(prompts)
    model._vision_reply = lambda instruction, images, tokens: "no json here"
    _generate(model, tmp_path, enhance_prompt=True)
    assert prompts == ["edit"]


def test_verify_retries_until_the_check_passes(tmp_path):
    model = _stub_model()
    replies = [
        '{"instruction_applied": false, "outside_unchanged": true}',
        '{"instruction_applied": true, "outside_unchanged": true}',
    ]
    model._vision_reply = lambda instruction, images, tokens: replies.pop(0)
    recorder = _Recorder()
    model.callbacks.register(recorder)
    image = _generate(model, tmp_path, verify=True, verify_retries=3)
    assert image.verification == {
        "verified": True,
        "instruction_applied": True,
        "outside_unchanged": True,
        "retries": 1,
    }
    assert len(model.transformer.calls) == 8  # original + one retry
    assert recorder.seeds == [1, 2]  # the retry moves to the next seed


def test_verify_retries_reuse_the_mask_and_rewritten_prompt(tmp_path):
    prompts, instructions = [], []
    model = _stub_model(prompts)
    replies = {
        "Outline": '[{"bbox_2d": [0, 0, 500, 1000], "label": "x"}]',
        "You": '{"rewritten_prompt": "a detailed edit"}',
    }
    verdicts = [
        '{"instruction_applied": false, "outside_unchanged": true}',
        '{"instruction_applied": true, "outside_unchanged": true}',
    ]

    def reply(instruction, images, tokens):
        instructions.append(instruction.split()[0])
        return verdicts.pop(0) if instruction.startswith("<image1> is the original") else replies[instructions[-1]]

    model._vision_reply = reply
    image = _generate(model, tmp_path, auto_mask="the left half", enhance_prompt=True, verify=True, verify_retries=2)
    assert image.verification["retries"] == 1
    assert instructions.count("Outline") == 1 and instructions.count("You") == 1  # grounding + rewrite once
    assert prompts == ["a detailed edit", "a detailed edit"]
    assert image.generation_parameters["auto_mask"] == "the left half"
    assert image.generation_parameters["original_prompt"] == "edit"
    pixels = np.asarray(image.image.convert("RGB")).astype(int)
    np.testing.assert_allclose(pixels[32, 56], SOURCE_RGB, atol=2)  # the retry kept the mask


@pytest.mark.parametrize("verify", [True, False])
def test_verify_survives_the_cli_memory_saver(tmp_path, verify):
    # The CLI always registers MemorySaver, which frees the text encoder before the loop
    # on single-seed runs; --verify reads the image back with it afterwards.
    model = _stub_model()
    model.processor = type(
        "Processor",
        (),
        {
            "__call__": staticmethod(
                lambda text, images, return_tensors: {
                    "input_ids": np.zeros((1, 3), np.int32),
                    "pixel_values": np.zeros((4, 8)),
                    "image_grid_thw": [[1, 2, 2]],
                }
            ),
            "tokenizer": SimpleNamespace(decode=lambda ids: '{"instruction_applied": true, "outside_unchanged": true}'),
            "image_processor": SimpleNamespace(size={"shortest_edge": 0, "longest_edge": 1e12}),
        },
    )()
    model.text_encoder = SimpleNamespace(generate=lambda *a, **k: [1])
    CallbackManager._register_memory_saver(Namespace(low_ram=False, seed=[42], verify=verify), model)
    if verify:
        assert _generate(model, tmp_path, verify=True).verification["verified"] is True
    else:
        # without --verify the saver still frees the encoder, which a verify call would need
        _generate(model, tmp_path)
        assert model.text_encoder is None


def test_verify_retries_survive_the_low_ram_memory_saver(tmp_path):
    # --low-ram frees the transformer after a single-seed loop; a retry regenerates with it.
    model = _stub_model()
    replies = [
        '{"instruction_applied": false, "outside_unchanged": true}',
        '{"instruction_applied": true, "outside_unchanged": true}',
    ]
    model._vision_reply = lambda instruction, images, tokens: replies.pop(0)
    previous_limit = mx.set_cache_limit(mx.device_info()["memory_size"])
    try:
        CallbackManager._register_memory_saver(Namespace(low_ram=True, seed=[42], verify=True, verify_retries=2), model)
        image = _generate(model, tmp_path, verify=True, verify_retries=2)
    finally:
        mx.set_cache_limit(previous_limit)
    assert image.verification["verified"] is True
    assert image.verification["retries"] == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"strength": 0.0}, "strength"),
        ({"strength": 1.5}, "strength"),
        ({"verify_retries": -1}, "verify_retries"),
    ],
)
def test_invalid_edit_arguments_are_rejected(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _generate(_stub_model(), tmp_path, **kwargs)


def test_edit_features_need_a_reference_image():
    with pytest.raises(ValueError, match="need a reference image"):
        _stub_model().generate_image(seed=1, prompt="p", num_inference_steps=4, auto_mask="x")


def test_vision_reply_expands_one_placeholder_per_image():
    model = _stub_model()
    seen = {}

    def processor(text, images, return_tensors):
        seen["text"], seen["sizes"] = text[0], [image.size for image in images]
        return {
            "input_ids": np.zeros((1, 3), np.int32),
            "pixel_values": np.zeros((4, 8)),
            "image_grid_thw": [[1, 2, 2]],
        }

    tokenizer = SimpleNamespace(decode=lambda ids: f"ids={ids}")
    model.processor = type("Processor", (), {"__call__": staticmethod(processor), "tokenizer": tokenizer})()
    model.text_encoder = SimpleNamespace(generate=lambda *a, **k: [5, 6])
    reply = model._vision_reply("compare", [Image.new("RGBA", (1024, 512)), Image.new("RGB", (64, 64))], 8)
    assert reply == "ids=[5, 6]"
    assert seen["text"].count("<|vision_start|><|image_pad|><|vision_end|>") == 2
    assert seen["text"].endswith("compare<|im_end|>\n<|im_start|>assistant\n")
    assert seen["sizes"][0] == (736, 352)  # ~512px budget, aspect kept


def test_step_cache_reuses_the_extract_step_on_an_unchanged_signal():
    mx.random.seed(0)
    model = QwenImage21Transformer(
        dict(
            num_layers=3,
            num_attention_heads=2,
            attention_head_dim=16,
            axes_dims_rope=(4, 6, 6),
            context_in_dim=32,
            in_channels=4,
            out_channels=4,
            mlp_ratio=3,
            eps=1e-6,
            causal_condition=True,
        )
    )
    layout = QwenImage21Layout.create(mx.array([False, True, True, False]), [(1, 2, 2)] * 3, (4, 6, 6))
    hidden, text = mx.random.normal((1, 12, 4)), mx.random.normal((1, 4, 32))
    cache, step_cache = [], StepCache(0.12)
    first = model(hidden, text, mx.array([0.5]), layout, cache, step_cache=step_cache)
    assert step_cache.hidden.shape[1] == layout.target_tokens
    stored = step_cache.hidden
    second = model(hidden, text, mx.array([0.5]), layout, cache, step_cache=step_cache)
    assert step_cache.hidden is stored  # skipped: blocks 1..N did not run
    np.testing.assert_allclose(np.array(second), np.array(first), rtol=1e-4, atol=1e-4)


def test_step_cache_recomputes_once_the_signal_accumulates():
    cache = StepCache(0.1)
    cache.store(mx.zeros((1, 2, 2)), mx.ones((1, 2, 2)))
    assert cache.should_skip(mx.ones((1, 2, 2)) * 1.05)
    assert not cache.should_skip(mx.ones((1, 2, 2)) * 1.2)


def test_generation_head_is_optional_when_loading():
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
    from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

    module = SimpleNamespace(
        parameters=lambda: {"lm_head": {"weight": mx.zeros((2, 2))}, "embed": {"weight": mx.zeros((2, 2))}},
        lm_head=object(),
        language_model=SimpleNamespace(norm=object()),
    )
    Qwen21Initializer._validate_weights(
        "text_encoder", module, {"embed.weight": mx.zeros((2, 2))}, QwenImage21WeightDefinition
    )
    assert module.has_generation_head is False
    # the random head is dropped, so a re-save cannot pass it off as real weights
    assert module.lm_head is None and module.language_model.norm is None
    with pytest.raises(ValueError, match="missing"):
        Qwen21Initializer._validate_weights(
            "vae", module, {"embed.weight": mx.zeros((2, 2))}, QwenImage21WeightDefinition
        )


def test_vision_replies_are_reused_across_seeds(tmp_path):
    model = _stub_model()
    calls = []
    model.processor = type(
        "Processor",
        (),
        {
            "__call__": staticmethod(
                lambda text, images, return_tensors: {
                    "input_ids": np.zeros((1, 3), np.int32),
                    "pixel_values": np.zeros((4, 8)),
                    "image_grid_thw": [[1, 2, 2]],
                }
            ),
            "tokenizer": SimpleNamespace(decode=lambda ids: '[{"bbox_2d": [0, 0, 500, 1000]}]'),
            "image_processor": SimpleNamespace(size={"shortest_edge": 0, "longest_edge": 1e12}),
        },
    )()
    model.text_encoder = SimpleNamespace(generate=lambda *a, **k: calls.append(1) or [1])
    for seed in (1, 2):
        _generate(model, tmp_path, auto_mask="the left half", seed=seed)
    assert len(calls) == 1
    other = Image.new("RGBA", (64, 64), (0, 0, 255, 255))
    model._vision_reply("locate", [other], 64)
    assert len(calls) == 2  # a different image is a new question


def test_grounding_parse_bbox_reads_0_1000_coordinates():
    parse = QwenImage21Grounding.parse_bbox
    assert parse('```json\n[{"bbox_2d": [100, 200, 400, 500], "label": "x"}]\n```') == (0.1, 0.2, 0.4, 0.5)
    assert parse("[[305, 234, 432, 372]]") == (0.305, 0.234, 0.432, 0.372)  # real reply, puffin beak
    assert parse("[[900, 900, 1020, 1010]]") == (0.9, 0.9, 1.0, 1.0)  # slight overshoot clamps
    assert parse("I cannot see it") is None
    assert parse("[[50, 50, 50, 90]]") is None  # degenerate
    assert parse("[[10, 20, 1100, 1700]]") is None  # far out of range


def test_grounding_parse_rewrite_and_verification():
    assert QwenImage21Grounding.parse_rewrite('noise {"rewritten_prompt": " x y "}') == "x y"
    assert QwenImage21Grounding.parse_rewrite('{"rewrited_prompt": "x y"}') == "x y"
    assert QwenImage21Grounding.parse_rewrite("nothing") is None
    verdict = '{"instruction_applied": true, "outside_unchanged": false}'
    assert QwenImage21Grounding.parse_verification(verdict) == (True, False)
    assert QwenImage21Grounding.parse_verification("no verdict") is None
    # the plain-language fallback (no parseable JSON) ignores spacing
    assert QwenImage21Grounding.parse_verification('"instruction_applied":true, "outside_unchanged" :true') == (
        True,
        True,
    )
    assert QwenImage21Grounding.parse_verification("instruction_applied: false, unchanged: true") == (False, True)
    # a field that only ends in applied or unchanged is not the verdict
    for reply in (
        "instruction_applied: false, partially_applied: true, outside_unchanged: true",
        "instruction_applied: false, not_applied: true, outside_unchanged: true",
    ):
        assert QwenImage21Grounding.parse_verification(reply) == (False, True)
    assert QwenImage21Grounding.parse_verification("instruction_applied: true, not_unchanged: true") == (True, False)


def test_grounding_masks():
    mask = QwenImage21Grounding.rasterize_mask((0.25, 0.25, 0.75, 0.75), (64, 64))
    assert mask.getpixel((32, 32)) == 255 and mask.getpixel((2, 2)) == 0
    grid = QwenImage21Grounding.to_latent_mask(mask, 4, 4)
    assert grid.shape == (4, 4) and grid[1, 1] > 0.9 and grid[0, 0] < 0.1


@pytest.mark.parametrize(
    "argv,match",
    [
        (["--auto-mask", "shirt"], "need at least one --image-paths"),
        (["--strength", "0"], "strength"),
        (["--verify-retries", "-1"], "verify-retries"),
        (["--verify-retries", "2"], "--verify-retries needs --verify"),
        (["--use-step-cache", "--no-use-kv-cache"], "--use-step-cache needs --use-kv-cache"),
    ],
)
def test_invalid_edit_flags_fail_before_loading(monkeypatch, capsys, argv, match):
    monkeypatch.setattr(sys, "argv", ["qwen21", "--prompt", "p", *argv])
    monkeypatch.setattr(cli, "QwenImage21Edit", lambda **kwargs: pytest.fail("Invalid input reached model loading"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert match in capsys.readouterr().err


def test_missing_mask_image_fails_before_loading(monkeypatch, tmp_path, capsys):
    source = _source(tmp_path)
    argv = ["qwen21", "--prompt", "p", "--image-paths", source, "--mask-image", str(tmp_path / "nope.png")]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli, "QwenImage21Edit", lambda **kwargs: pytest.fail("Invalid input reached model loading"))
    with pytest.raises(SystemExit):
        cli.main()
    assert "--mask-image not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "ids,grids",
    [
        ([1, 2, 3, 4, 5], None),
        ([1, 98, 99, 99, 99, 99, 97, 5], [[1, 4, 4]]),
    ],
)
def test_generate_matches_torch_greedy_decoding(ids, grids):
    config = dict(
        image_token_id=99,
        vision_start_token_id=98,
        vision_end_token_id=97,
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=64,
            head_dim=16,
            max_position_embeddings=2048,
            rope_theta=5000000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            rope_scaling=dict(rope_type="default", mrope_section=[4, 2, 2], mrope_interleaved=True),
        ),
        vision_config=dict(
            depth=3,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1, 2],
            hidden_act="gelu_pytorch_tanh",
        ),
    )
    torch.manual_seed(7)
    reference = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config)).to(device="cpu", dtype=torch.float32).eval()
    with torch.no_grad():
        reference.lm_head.weight.mul_(20)  # sharpen logits so greedy ties cannot flip on rounding
    model = QwenImage21TextEncoder(config)
    weights = []
    for key, value in reference.state_dict().items():
        mapped = QwenImage21WeightDefinition.text_key(key)
        weights.append((mapped, QwenImage21WeightDefinition.text_weight(mapped, mx.array(value.numpy()))))
    model.load_weights(weights, strict=False)  # rotary inv_freq buffers are computed, not loaded
    input_ids = torch.tensor([ids])
    kwargs, mlx_kwargs = {}, {}
    if grids is not None:
        grid = torch.tensor(grids)
        pixels = torch.randn(int(grid.prod(-1).sum()), 24)
        kwargs = dict(pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(input_ids == 99).int())
        mlx_kwargs = dict(pixel_values=mx.array(pixels.numpy()), image_grid_thw=mx.array(grids))
    with torch.no_grad():
        expected = reference.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            **kwargs,
        )[0, len(ids) :].tolist()
    actual = model.generate(mx.array([ids]), max_new_tokens=6, stop_token_ids=(), **mlx_kwargs)
    assert actual == expected
