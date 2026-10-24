import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.fibo2 import fibo2_initializer
from mflux.models.fibo2.fibo2_initializer import Fibo2Initializer
from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.model.fibo2_text_encoder.prompt_encoder import Fibo2PromptEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.variants import fibo2_core
from mflux.models.fibo2.variants.fibo2_core import _Fibo2Core
from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition

pytestmark = pytest.mark.fast


class _Core:
    # A _Fibo2Core without weights, for the methods that only need its config and a few attributes
    @staticmethod
    def of(model_config: ModelConfig, **attributes) -> _Fibo2Core:
        core = _Fibo2Core.__new__(_Fibo2Core)
        nn.Module.__init__(core)
        core.model_config = model_config
        for name, value in attributes.items():
            setattr(core, name, value)
        return core


class _Loop:
    # Stands in for the encoders, the transformer and the decoder, and records the embeddings each step is given
    WITH_TEXT_IMAGES = mx.ones((1,))
    WITHOUT_TEXT_IMAGES = mx.zeros((1,))

    def __init__(self):
        self.embeds_per_step: list[str] = []
        self.encoded: list[str] = []

    def encode_prompts(self, prompt, images, text_encoder_images, render_text_images):
        return _Loop.WITH_TEXT_IMAGES, _Loop.WITHOUT_TEXT_IMAGES

    def encode_prompt(self, prompt, images=None):
        self.encoded.append(prompt)
        return mx.zeros((1,))

    def predict(self, latents, sigma, embeds, negative_embeds, guidance, context_latents):
        self.embeds_per_step.append("with" if embeds is _Loop.WITH_TEXT_IMAGES else "without")
        return mx.zeros_like(latents)

    @staticmethod
    def decode(vae, latents, tiling_config=None):
        return latents

    @staticmethod
    def to_image(**kwargs):
        return kwargs

    def core(self, monkeypatch, model_config: ModelConfig | None = None, vae=None, stand_in_prompts: bool = True):
        # A core without weights whose loop runs on these stand-ins. Without stand_in_prompts the prompts go through
        # the real _encode_prompts, and only the text encoder is a stand-in.
        core = _Core.of(model_config or ModelConfig.fibo2(), callbacks=CallbackRegistry(), vae=vae, tiling_config=None, bits=None, fonts_dir=None)  # fmt: off
        core.lora_paths, core.lora_scales = None, None
        if stand_in_prompts:
            core._encode_prompts = self.encode_prompts
        core._encode_prompt = self.encode_prompt
        core._predict = self.predict
        monkeypatch.setattr(Fibo2LatentCreator, "decode", staticmethod(_Loop.decode))
        monkeypatch.setattr(Fibo2LatentCreator, "encode_image", staticmethod(_Loop.encode_image))
        monkeypatch.setattr(fibo2_core.ImageUtil, "to_image", staticmethod(_Loop.to_image))
        return core

    def generate(self, monkeypatch, model_config: ModelConfig | None = None, vae=None, **kwargs) -> dict:
        # Returns what reaches ImageUtil.to_image
        settings = {"seed": 1, "prompt": "p", "num_inference_steps": 8, "height": 512, "width": 512, "guidance": 5.0}
        return self.core(monkeypatch, model_config, vae)._generate(**{**settings, **kwargs})

    @staticmethod
    def encode_image(vae, image, tiling_config=None):
        return mx.zeros((1, 32, image.height // 8, image.width // 8))


class _Fonts:
    # Records the patterns PathResolution is asked for
    def __init__(self, root: Path):
        self.root = root
        self.patterns: list[list[str]] = []

    def resolve(self, path, patterns=None, saved_subdirs=None):
        self.patterns.append(list(patterns or []))
        return self.root


class _Processor:
    @staticmethod
    def save_pretrained(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "preprocessor_config.json").write_text("{}")


class _Saver:
    @staticmethod
    def save_model(**kwargs) -> None:
        return None


@pytest.mark.parametrize(
    ("name", "base_model", "expected"),
    [
        ("fibo2", None, (50, 5.0)),
        ("fibo2-edit-turbo", None, (4, 1.0)),
        ("/models/my-own-turbo", "fibo2-edit-turbo", (4, 1.0)),
        ("/models/my-own-dev", "fibo2", (50, 5.0)),
    ],
)
def test_steps_and_guidance_default_by_checkpoint(name, base_model, expected):
    core = _Core.of(ModelConfig.from_name(name, base_model=base_model))
    assert core._resolve_sampling(None, None) == expected
    assert core._resolve_sampling(12, 3.0) == (12, 3.0)


def test_the_loop_switches_to_the_text_images_once_the_noise_level_is_at_most_0_9(monkeypatch):
    # The schedule diffusers runs for 8 steps at 512x512: sigmas 1.0 and 0.929 run without the text images
    loop = _Loop()
    loop.generate(monkeypatch)
    assert loop.embeds_per_step == ["without", "without"] + ["with"] * 6


@pytest.mark.parametrize(
    ("decode_mode", "recorded"),
    [(VAEBlockDecoder.LOW_MEMORY, {}), (VAEBlockDecoder.FULL_PRECISION, {"vae_full_precision": True})],
)
def test_the_metadata_records_a_full_precision_decode(monkeypatch, decode_mode, recorded):
    image = _Loop().generate(monkeypatch, vae=SimpleNamespace(decode_mode=decode_mode))
    assert image["generation_parameters"] == recorded


@pytest.mark.parametrize(
    ("prompt", "images"),
    [
        ("Make the bicycle blue", 1),
        (json.dumps({"structured_caption": "A blue bicycle.", "edit_instruction": "Make the bicycle blue"}), 1),
        (json.dumps({"short_description": "A blue bicycle.", "edit_instruction": "Make the bicycle blue"}), 2),
        (json.dumps({"short_description": "A blue bicycle.", "lighting": {}}), 0),
    ],
    ids=["instruction", "structured-caption", "caption-two-images", "text-to-image"],
)
def test_callbacks_and_metadata_keep_the_prompt_as_given(monkeypatch, prompt, images):
    # Only the text encoder reads the prompt with its image markers. With the markers in the metadata, a replay of a
    # "structured_caption" edit failed check_edit_prompt, and an instruction alone gained a second image marker.
    loop = _Loop()
    core = loop.core(monkeypatch, stand_in_prompts=False)
    started = []
    start = core.callbacks.start
    core.callbacks.start = lambda **kwargs: started.append(kwargs["prompt"]) or start(**kwargs)
    crops = [Image.new("RGB", (64, 64))] * images
    settings = {"seed": 1, "num_inference_steps": 2, "height": 64, "width": 64, "guidance": 1.0, "images": crops, "text_encoder_images": crops}  # fmt: off
    recorded = core._generate(prompt=prompt, **settings)["prompt"]
    replayed = core._generate(prompt=recorded, **settings)["prompt"]
    assert recorded == replayed == prompt
    assert started == [prompt, prompt]
    assert loop.encoded[0] == loop.encoded[1] != prompt
    if images:
        Fibo2PromptEncoder.check_edit_prompt(recorded)
        assert loop.encoded[0].count(Fibo2PromptEncoder.VISION_MARKER) == images


@pytest.mark.parametrize(
    ("negative_prompt", "images", "encoded"),
    [
        (None, 0, Fibo2PromptEncoder.TEXT_TO_IMAGE_NEGATIVE),
        ("", 1, Fibo2PromptEncoder.NULL_CAPTION),
        ("blurry", 0, "blurry"),
        ("blurry", 1, "blurry"),
    ],
)
def test_the_negative_prompt_defaults_to_the_model_cards(monkeypatch, negative_prompt, images, encoded):
    # Text to image steers away from the card's caption of soft, featureless images, an edit from the null caption
    loop = _Loop()
    crops = [Image.new("RGB", (64, 64))] * images
    loop.generate(monkeypatch, negative_prompt=negative_prompt, images=crops, text_encoder_images=crops)
    assert loop.encoded == [encoded]


def test_the_text_to_image_negative_has_the_keys_of_the_null_caption():
    negative = json.loads(Fibo2PromptEncoder.TEXT_TO_IMAGE_NEGATIVE)
    assert list(negative) == list(json.loads(Fibo2PromptEncoder.NULL_CAPTION))
    assert negative["objects"][0]["description"] == "the main subject"


def test_a_step_cache_reuses_the_velocity_on_the_steps_it_skips(monkeypatch):
    # 20 steps at 0.25: the 5 steps whose sigma changes least, outside the first and last 2, skip the transformer
    loop = _Loop()
    image = loop.generate(monkeypatch, num_inference_steps=20, step_cache_ratio=0.25)
    assert len(loop.embeds_per_step) == 15
    assert image["generation_parameters"] == {"step_cache_ratio": 0.25}
    loop = _Loop()
    assert loop.generate(monkeypatch, num_inference_steps=20)["generation_parameters"] == {}
    assert len(loop.embeds_per_step) == 20


@pytest.mark.parametrize(("render_text_images", "recorded"), [(True, {}), (False, {"render_text_images": False})])
def test_the_metadata_records_text_images_that_are_off(monkeypatch, render_text_images, recorded):
    image = _Loop().generate(monkeypatch, render_text_images=render_text_images)
    assert image["generation_parameters"] == recorded


def test_save_model_writes_the_processor_config_and_the_fonts_with_their_licences(tmp_path, monkeypatch):
    fonts = tmp_path / "source_fonts"
    fonts.mkdir()
    for name in ("Aileron-Regular.otf", "DejaVuSans.ttf", "LICENSE-DejaVu.txt"):
        (fonts / name).write_bytes(b"x")
    monkeypatch.setattr(ModelSaver, "save_model", staticmethod(_Saver.save_model))
    processor = SimpleNamespace(image_processor=_Processor)
    _Core.of(ModelConfig.fibo2(), bits=None, processor=processor, fonts_dir=fonts).save_model(str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "processor" / "preprocessor_config.json").is_file()
    assert sorted(p.name for p in (tmp_path / "saved" / "fonts").iterdir()) == [
        "Aileron-Regular.otf",
        "DejaVuSans.ttf",
        "LICENSE-DejaVu.txt",
    ]


def test_fonts_are_resolved_on_their_own_with_their_licences(tmp_path, monkeypatch):
    # A snapshot cached without fonts/ counts as complete for the weights, so the fonts must be asked for by name
    (tmp_path / "fonts").mkdir()
    fonts = _Fonts(tmp_path)
    monkeypatch.setattr(fibo2_initializer.PathResolution, "resolve", fonts.resolve)
    assert Fibo2Initializer._fonts_dir("briaai/Fibo-2.0-edit-turbo") == tmp_path / "fonts"
    assert fonts.patterns == [Fibo2WeightDefinition.FONT_PATTERNS]
    assert "fonts/LICENSE-*.txt" in Fibo2WeightDefinition.FONT_PATTERNS
    assert set(Fibo2WeightDefinition.FONT_PATTERNS) <= set(Fibo2WeightDefinition.get_download_patterns())


def test_missing_fonts_are_reported(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(fibo2_initializer.PathResolution, "resolve", _Fonts(tmp_path).resolve)
    assert Fibo2Initializer._fonts_dir("some/model") is None
    assert "fonts were not found" in caplog.text


@pytest.mark.parametrize(("text_length", "gist"), [(0, 256), (256, 256), (257, 320), (1000, 384), (2048, 512), (5000, 512)])  # fmt: off
def test_gist_count_follows_the_text_length(text_length, gist):
    # Values from diffusers' compute_gist_count with the checkpoint's config
    transformer = Fibo2Transformer(dim=128, n_heads=2, n_layers=1, cap_feat_dim=128, injection_layer_ids=(0,), axes_dims=(16, 24, 24))  # fmt: off
    assert transformer.perceiver.gist_count(text_length) == gist


def test_the_transformer_reads_each_context_image_in_its_place():
    mx.random.seed(0)
    transformer = Fibo2Transformer(dim=128, n_heads=2, n_layers=2, cap_feat_dim=96, injection_layer_ids=(1,), axes_dims=(16, 24, 24))  # fmt: off
    latents = mx.random.normal((1, 32, 16, 24), key=mx.random.key(0))
    text = mx.random.normal((1, 2, 300, 96), key=mx.random.key(1))
    first = mx.random.normal((1, 32, 8, 8), key=mx.random.key(2))
    second = mx.random.normal((1, 32, 12, 20), key=mx.random.key(3))
    other = mx.random.normal((1, 32, 12, 20), key=mx.random.key(4))
    sigma = mx.array([0.7])

    both = np.array(transformer(latents, sigma, text, context_latents=[first, second]))
    assert both.shape == latents.shape
    np.testing.assert_array_equal(both, np.array(transformer(latents, sigma, text, context_latents=[first, second])))
    for changed in ([first, other], [second, first], [first], None):
        assert not np.allclose(both, np.array(transformer(latents, sigma, text, context_latents=changed)))


def test_each_context_image_sits_on_its_own_rope_plane_after_the_image():
    # diffusers: the image on plane max_num_gist_tokens + 1 (513), context image k on plane 513 + k + 1, each with its
    # own rows and columns from 0
    transformer = Fibo2Transformer(dim=128, n_heads=2, n_layers=1, cap_feat_dim=128, injection_layer_ids=(0,), axes_dims=(16, 24, 24))  # fmt: off
    latents, first, second = mx.zeros((1, 32, 4, 6)), mx.zeros((1, 32, 2, 2)), mx.zeros((1, 32, 6, 4))
    ids = np.array(transformer.image_position_ids(latents, [first, second]))
    planes = [513] * (2 * 3) + [514] * (1 * 1) + [515] * (3 * 2)
    assert ids[:, 0].tolist() == planes
    assert ids[:6, 1:].tolist() == [[row, col] for row in range(2) for col in range(3)]
    assert ids[7:, 1:].tolist() == [[row, col] for row in range(3) for col in range(2)]


def test_a_diffusers_named_lora_applies_to_renamed_and_injection_layers(tmp_path):
    transformer = Fibo2Transformer(dim=128, n_heads=2, n_layers=2, cap_feat_dim=128, injection_layer_ids=(1,), axes_dims=(16, 24, 24))  # fmt: off
    modules = {
        "t_embedder.mlp.0": transformer.t_embedder.linear1,
        "final_layer.adaLN_modulation.1": transformer.final_layer.adaLN_modulation[0],
        "layers.1.cross_attention.to_k": transformer.layers[1].cross_attention.to_k,
        "perceiver.layers.0.ffn.w2": transformer.perceiver.layers[0].ffn.w2,
    }
    weights = {}
    for name, linear in modules.items():
        out_dim, in_dim = linear.weight.shape
        weights[f"transformer.{name}.lora_A.weight"] = mx.ones((4, in_dim))
        weights[f"transformer.{name}.lora_B.weight"] = mx.ones((out_dim, 4))
    path = tmp_path / "pytorch_lora_weights.safetensors"
    mx.save_safetensors(str(path), weights)
    LoRALoader.load_and_apply_lora(Fibo2LoRAMapping.get_mapping(), transformer, [str(path)], [1.0], bake_lora=False)
    adapted = [name for name, module in transformer.named_modules() if isinstance(module, LoRALinear)]
    assert sorted(adapted) == sorted(
        [
            "t_embedder.linear1",
            "final_layer.adaLN_modulation.0",
            "layers.1.cross_attention.to_k",
            "perceiver.layers.0.ffn.w2",
        ]  # fmt: off
    )


def test_a_name_that_only_starts_like_fibo_2_stays_with_fibo():
    assert ModelConfig.from_name("me/fibo-2025-finetune").base_model == ModelConfig.fibo().model_name
