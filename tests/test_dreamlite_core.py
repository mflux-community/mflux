from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.dreamlite.latent_creator.dreamlite_latent_creator import DreamLiteLatentCreator
from mflux.models.dreamlite.model.dreamlite_text_encoder.prompt_encoder import DreamLitePromptEncoder
from mflux.models.dreamlite.model.dreamlite_unet.unet import DreamLiteUNet
from mflux.models.dreamlite.variants import dreamlite_core
from mflux.models.dreamlite.variants.dreamlite_core import _DreamLiteCore
from mflux.models.dreamlite.variants.edit.dreamlite_edit import DreamLiteEdit
from mflux.models.dreamlite.variants.txt2img.dreamlite import DreamLite

pytestmark = pytest.mark.fast
mx_compile = mx.compile


@pytest.fixture(autouse=True)
def uncompiled_predict(monkeypatch):
    # The stand-in UNet reads values out of its inputs, which a compiled function cannot do
    monkeypatch.setattr(dreamlite_core.AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: True))


class _Core:
    # A DreamLite model without weights: the UNet, the encoders and the VAE are stand-ins that record their calls
    @staticmethod
    def of(cls, model_config: ModelConfig) -> "_DreamLiteCore":
        core = cls.__new__(cls)
        nn.Module.__init__(core)
        core.model_config = model_config
        core.prompt_cache = {}
        core.callbacks = CallbackRegistry()
        core.tiling_config = None
        core.bits = None
        core.compute_precision = ComputePrecision(None)
        core.calls = []
        core.encoded = []
        core.unet = _Core._unet(core)
        core.vae = SimpleNamespace(
            decode=lambda latents: mx.zeros((1, 3, latents.shape[2] * 8, latents.shape[3] * 8)),
            encode=lambda image: mx.full((1, 4, image.shape[2] // 8, image.shape[3] // 8), 3.0),
        )
        core._encode_generate = lambda text: _Core._encode(core, ("generate", text))
        core._encode_edit = lambda text, image, vision_size: _Core._encode(core, ("edit", text, vision_size))
        return core

    @staticmethod
    def _encode(core, call) -> mx.array:
        core.encoded.append(call)
        return mx.full((1, 2, 4), float(len(core.encoded)))

    @staticmethod
    def _unet(core):
        def unet(model_input, timestep, prompt_embeds, size):
            width = model_input.shape[-1] // 2
            core.calls.append(
                {
                    "embeds": prompt_embeds[0, 0, 0].item(),
                    "context": model_input[0, 0, 0, width].item(),
                    "shape": model_input.shape,
                    "size": size,
                }
            )
            # The right half must never reach the scheduler
            return mx.concatenate([mx.ones_like(model_input[..., :width]), mx.full(model_input[..., width:].shape, 99.0)], axis=3)  # fmt: skip

        return unet


def test_mobile_runs_one_pass_per_step_on_latents_beside_an_empty_context():
    core = _Core.of(DreamLite, ModelConfig.dreamlite_mobile())

    image = core.generate_image(seed=1, prompt="a puffin", width=128, height=64)

    assert core.encoded == [("generate", "[Generate]: a puffin")]
    assert len(core.calls) == 4
    assert {call["shape"] for call in core.calls} == {(1, 4, 8, 32)}
    assert {call["context"] for call in core.calls} == {0.0}
    assert {call["size"] for call in core.calls} == {(128, 64)}
    assert image.image.size == (128, 64)
    assert (image.steps, image.guidance) == (4, 1.0)


def test_base_runs_the_negative_prompt_as_a_second_pass():
    core = _Core.of(DreamLite, ModelConfig.dreamlite())

    image = core.generate_image(seed=1, prompt="a puffin", num_inference_steps=3, width=64, height=64, negative_prompt="blur")  # fmt: skip

    assert core.encoded == [("generate", "[Generate]: a puffin"), ("generate", "blur")]
    assert [call["embeds"] for call in core.calls] == [1.0, 2.0] * 3
    assert (image.steps, image.guidance, image.negative_prompt) == (3, 3.5, "blur")


def test_the_default_negative_prompt_is_empty_and_has_no_generate_prefix():
    core = _Core.of(DreamLite, ModelConfig.dreamlite())

    core.generate_image(seed=1, prompt="a puffin", num_inference_steps=1, width=64, height=64)

    assert core.encoded[1] == ("generate", "")


def test_mobile_takes_no_guidance_like_the_reference_pipeline():
    core = _Core.of(DreamLite, ModelConfig.dreamlite_mobile())

    image = core.generate_image(seed=1, prompt="a puffin", width=64, height=64, guidance=5.0, negative_prompt="blur")

    assert core.encoded == [("generate", "[Generate]: a puffin")]
    assert len(core.calls) == 4
    assert image.guidance == 1.0


def test_guidance_at_one_skips_the_negative_pass_on_the_base_model():
    core = _Core.of(DreamLite, ModelConfig.dreamlite())

    core.generate_image(seed=1, prompt="a puffin", num_inference_steps=2, width=64, height=64, guidance=1.0)

    assert core.encoded == [("generate", "[Generate]: a puffin")]
    assert len(core.calls) == 2


def test_prompt_encodings_are_cached_between_seeds_and_edits_are_not():
    core = _Core.of(DreamLite, ModelConfig.dreamlite())
    for seed in (1, 2):
        core.generate_image(seed=seed, prompt="a puffin", num_inference_steps=1, width=64, height=64)

    assert len(core.encoded) == 2
    assert "a puffin" in core.prompt_cache

    edit = _Core.of(DreamLiteEdit, ModelConfig.dreamlite_mobile())
    image = Image.new("RGB", (64, 64))
    for seed in (1, 2):
        edit._generate(seed=seed, prompt="x", num_inference_steps=1, height=64, width=64, guidance=None, image=image)
    assert len(edit.encoded) == 2
    assert edit.prompt_cache == {}


def test_an_edit_runs_on_a_square_copy_and_returns_the_source_size(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (96, 48), "red").save(source)
    core = _Core.of(DreamLiteEdit, ModelConfig.dreamlite_mobile())

    image = core.generate_image(seed=1, prompt="make it snowy", image_path=source, num_inference_steps=1)

    assert core.encoded == [("edit", DreamLitePromptEncoder.EDIT_PREFIX + "make it snowy", 256)]
    assert core.calls[0]["shape"] == (1, 4, 128, 256)
    assert core.calls[0]["size"] == (1024, 1024)
    assert core.calls[0]["context"] == 3.0
    assert image.image.size == (96, 48)
    assert (image.width, image.height) == (96, 48)
    assert image.image_paths == [source]


def test_an_edit_can_be_saved_at_another_size(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (96, 48)).save(source)
    core = _Core.of(DreamLiteEdit, ModelConfig.dreamlite_mobile())

    image = core.generate_image(seed=1, prompt="x", image_path=source, num_inference_steps=1, width=192, height=64)

    assert image.image.size == (192, 64)


def test_a_base_edit_shows_qwen_the_larger_copy_and_keeps_the_image_in_both_passes(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64)).save(source)
    core = _Core.of(DreamLiteEdit, ModelConfig.dreamlite())

    core.generate_image(seed=1, prompt="x", image_path=source, num_inference_steps=1)

    assert [call[2] for call in core.encoded] == [512, 512]
    assert core.encoded[1][1] == ""
    assert [(call["embeds"], call["context"]) for call in core.calls] == [(1.0, 3.0), (2.0, 3.0)]


def test_image_guidance_adds_a_pass_without_the_source_image(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64)).save(source)
    core = _Core.of(DreamLiteEdit, ModelConfig.dreamlite())

    image = core.generate_image(seed=1, prompt="x", image_path=source, num_inference_steps=1, image_guidance=1.5)

    assert [(call["embeds"], call["context"]) for call in core.calls] == [(1.0, 3.0), (2.0, 3.0), (2.0, 0.0)]
    assert image.generation_parameters == {"image_guidance": 1.5}


def test_guidance_combines_the_passes_as_the_reference_does():
    # uncond + g * (text - image) + ig * (image - uncond), with the three passes told apart by their inputs
    text, image, uncond = 5.0, 2.0, -1.0
    latents, context, empty = mx.zeros((1, 4, 2, 2)), mx.ones((1, 4, 2, 2)), mx.zeros((1, 4, 2, 2))
    prompt_embeds, negative_embeds = mx.ones((1, 1, 4)), mx.zeros((1, 1, 4))
    passes = []

    def unet(model_input, timestep, embeds, size):
        has_image = model_input[0, 0, 0, 2].item() == 1.0
        name = "text" if embeds is prompt_embeds else ("image" if has_image else "uncond")
        passes.append(name)
        return mx.full(model_input.shape, {"text": text, "image": image, "uncond": uncond}[name])

    def guide(image_guidance):
        predict = _DreamLiteCore._predict(unet, prompt_embeds, negative_embeds, context, empty, (16, 16), 3.5, image_guidance)  # fmt: skip
        return predict(latents, mx.array(1.0))

    assert np.isclose(guide(1.0)[0, 0, 0, 0].item(), uncond + 3.5 * (text - image) + 1.0 * (image - uncond))
    assert np.isclose(guide(1.5)[0, 0, 0, 0].item(), uncond + 3.5 * (text - image) + 1.5 * (image - uncond))
    assert np.isclose(guide(None)[0, 0, 0, 0].item(), image + 3.5 * (text - image))
    # Only an image guidance other than 1.0 pays for the pass without the source image
    assert passes == ["text", "image", "text", "image", "uncond", "text", "image"]
    assert guide(1.0).shape == latents.shape


def test_the_compiled_predict_gives_what_the_plain_one_gives(monkeypatch):
    # The compiled function closes over the embeddings and the context image; a real (tiny) UNet, three passes
    unet = DreamLiteUNet(block_channels=(64, 128, 128), num_heads=(1, 2, 2), transformer_layers=(1, 1, 1), time_proj_dim=64, time_embed_dim=128, text_dim=128, context_dim=128)  # fmt: skip
    keys = [mx.random.key(index) for index in range(5)]
    latents, context = mx.random.normal((1, 4, 8, 8), key=keys[0]), mx.random.normal((1, 4, 8, 8), key=keys[1])
    prompt_embeds, negative_embeds = (
        mx.random.normal((1, 5, 128), key=keys[2]),
        mx.random.normal((1, 3, 128), key=keys[3]),
    )
    arguments = (unet, prompt_embeds, negative_embeds, context, mx.zeros_like(context), (64, 64), 3.5, 1.5)

    plain = _DreamLiteCore._predict(*arguments)(latents, mx.array(500.0))
    compiled = []
    monkeypatch.setattr(dreamlite_core.AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    monkeypatch.setattr(mx, "compile", lambda function: compiled.append(function) or mx_compile(function))
    fast = _DreamLiteCore._predict(*arguments)(latents, mx.array(500.0))

    assert len(compiled) == 1
    assert fast.dtype == mx.float32 and fast.shape == latents.shape
    assert mx.allclose(plain, fast, atol=1e-4).item()


def test_the_run_records_its_compute_precision():
    core = _Core.of(DreamLite, ModelConfig.dreamlite_mobile())
    core.compute_precision = ComputePrecision(mx.float16)

    image = core.generate_image(seed=1, prompt="a puffin", num_inference_steps=1, width=64, height=64)

    assert image.generation_parameters == {"compute_precision": "float16"}


def test_sizes_must_be_multiples_of_32():
    core = _Core.of(DreamLite, ModelConfig.dreamlite_mobile())

    with pytest.raises(ValueError, match="multiples of 32"):
        core.generate_image(seed=1, prompt="x", width=1000, height=1024)
    with pytest.raises(ValueError, match="must be positive"):
        DreamLiteLatentCreator.check_size(height=0, width=64)
    DreamLiteLatentCreator.check_size(height=768, width=1280)
    assert core.calls == []


def test_the_noise_is_seeded_float32_at_an_eighth_of_the_image():
    first = DreamLiteLatentCreator.create_noise(3, height=64, width=96)

    assert first.shape == (1, 4, 8, 12)
    assert first.dtype == mx.float32
    assert mx.array_equal(first, DreamLiteLatentCreator.create_noise(3, height=64, width=96)).item()
    assert not mx.array_equal(first, DreamLiteLatentCreator.create_noise(4, height=64, width=96)).item()


def test_the_context_sits_to_the_right_of_the_latents_and_tiled_latents_are_squeezed():
    latents, context = mx.zeros((1, 4, 2, 3)), mx.ones((1, 4, 2, 3))

    joined = DreamLiteLatentCreator.with_context(latents, context)

    assert joined.shape == (1, 4, 2, 6)
    assert joined[0, 0, 0, 2].item() == 0.0 and joined[0, 0, 0, 3].item() == 1.0
    assert DreamLiteLatentCreator.pack_latents(mx.zeros((1, 4, 1, 2, 3))).shape == (1, 4, 2, 3)


def test_the_source_image_is_encoded_at_the_edit_size():
    seen = []
    vae = SimpleNamespace(encode=lambda image: seen.append(image) or mx.zeros((1, 4, 4, 8)))

    latents = DreamLiteLatentCreator.encode_image(vae, Image.new("RGB", (30, 50), "white"), height=32, width=64)

    assert seen[0].shape == (1, 3, 32, 64)
    assert np.isclose(seen[0].max().item(), 1.0)
    assert latents.dtype == mx.float32


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, text, images=None, **kwargs):
        self.calls.append((text, images, kwargs))
        inputs = {"input_ids": np.arange(70)[None]}
        if images:
            inputs.update(pixel_values=np.zeros((4, 8)), image_grid_thw=np.array([[1, 2, 2]]))
        return inputs


class _FakeTextEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, pixel_values=None, image_grid_thw=None):
        self.calls.append((input_ids, pixel_values, image_grid_thw))
        return mx.arange(input_ids.shape[1]).astype(mx.float32)[None, :, None]


def test_a_generation_prompt_goes_through_its_template_and_drops_the_system_part():
    processor, text_encoder = _FakeProcessor(), _FakeTextEncoder()

    embeds = DreamLitePromptEncoder.encode_generate(DreamLitePromptEncoder.generate_prompt("a puffin"), processor, text_encoder)  # fmt: skip

    text, images, _ = processor.calls[0]
    assert text[0].startswith("<|im_start|>system\nDescribe the image by detailing")
    assert text[0].endswith("<|im_start|>user\n[Generate]: a puffin<|im_end|>\n<|im_start|>assistant\n")
    assert images is None
    assert text_encoder.calls[0][1] is None
    assert embeds.shape == (1, 70 - 34, 1)
    assert embeds[0, 0, 0].item() == 34.0


def test_an_edit_prompt_carries_the_image_marker_and_a_square_copy_of_the_source():
    processor, text_encoder = _FakeProcessor(), _FakeTextEncoder()

    embeds = DreamLitePromptEncoder.encode_edit(DreamLitePromptEncoder.edit_prompt("make it snowy"), Image.new("RGB", (40, 20)), 256, processor, text_encoder)  # fmt: skip

    text, images, _ = processor.calls[0]
    assert "<|vision_start|><|image_pad|><|vision_end|>[Edit]: A diptych with two side-by-side images" in text[0]
    assert text[0].endswith("the left one has make it snowy<|im_end|>\n<|im_start|>assistant\n")
    assert images[0].size == (256, 256)
    assert text_encoder.calls[0][1] is not None and text_encoder.calls[0][2] is not None
    assert embeds.shape == (1, 70 - 64, 1)
