from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import PIL.Image
import pytest
from mlx import nn

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.boogu.variants import BooguImage
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mflux.models.ideogram4.variants import Ideogram4
from mflux.models.krea2.variants import Krea2
from mflux.models.lens.variants.txt2img.lens_image import LensImage


class _FakeQwen3Tokenizer:
    def tokenize(self, prompt, max_length=None):
        length = max(len(prompt), 1)
        input_ids = mx.arange(length, dtype=mx.int32)[None, :]
        attention_mask = mx.ones((1, length), dtype=mx.int32)
        return type("TokenizerOutput", (), {"input_ids": input_ids, "attention_mask": attention_mask})()


class _FakeQwen3TextEncoder:
    def get_prompt_embeds(self, input_ids, attention_mask, hidden_state_layers):
        return mx.zeros((1, input_ids.shape[1], 8), dtype=mx.float32)


class _FakeHiddenStatesTransformer:
    def __call__(self, **kwargs):
        return mx.zeros_like(kwargs["hidden_states"])


class _FakeLensTextEncoder:
    def encode(self, prompt):
        return mx.zeros((1, 4, 8))


class _RecordingPackedVAE:
    def __init__(self):
        self.received_tiling_configs: list[TilingConfig | None] = []
        self.bn = SimpleNamespace(running_mean=mx.zeros((128,)), running_var=mx.ones((128,)), eps=1e-4)

    def decode_packed_latents(self, packed_latents, tiling_config=None):
        self.received_tiling_configs.append(tiling_config)
        return mx.zeros((packed_latents.shape[0], 3, packed_latents.shape[2] * 8, packed_latents.shape[3] * 8))

    def encode(self, image):
        height, width = image.shape[-2], image.shape[-1]
        return mx.zeros((1, 32, height // 8, width // 8))


class _PackedModelFixtures:
    @staticmethod
    def flux2_klein(vae: _RecordingPackedVAE, tiling_config: TilingConfig | None) -> Flux2Klein:
        model = Flux2Klein.__new__(Flux2Klein)
        model.model_config = ModelConfig.flux2_klein_4b()
        model.callbacks = CallbackRegistry()
        model.tokenizers = {"qwen3": _FakeQwen3Tokenizer()}
        model.text_encoder = _FakeQwen3TextEncoder()
        model.transformer = _FakeHiddenStatesTransformer()
        model.vae = vae
        model.bits = None
        model.lora_paths = None
        model.lora_scales = None
        model.tiling_config = tiling_config
        return model

    @staticmethod
    def flux2_klein_edit(vae: _RecordingPackedVAE, tiling_config: TilingConfig | None) -> Flux2KleinEdit:
        model = Flux2KleinEdit.__new__(Flux2KleinEdit)
        model.model_config = ModelConfig.flux2_klein_4b()
        model.callbacks = CallbackRegistry()
        model.tokenizers = {"qwen3": _FakeQwen3Tokenizer()}
        model.text_encoder = _FakeQwen3TextEncoder()
        model.transformer = _FakeHiddenStatesTransformer()
        model.vae = vae
        model.bits = None
        model.lora_paths = None
        model.lora_scales = None
        model.tiling_config = tiling_config
        return model

    @staticmethod
    def lens_image(vae: _RecordingPackedVAE, tiling_config: TilingConfig | None) -> LensImage:
        model = LensImage.__new__(LensImage)
        model.model_config = ModelConfig.lens_turbo()
        model.callbacks = CallbackRegistry()
        model.text_encoder = _FakeLensTextEncoder()
        model.transformer = _FakeHiddenStatesTransformer()
        model.vae = vae
        model.bits = None
        model.tiling_config = tiling_config
        return model

    @staticmethod
    def write_reference_image(path) -> None:
        PIL.Image.new("RGB", (64, 64), color=(128, 128, 128)).save(path)


class _FakeKrea2Tokenizer:
    def tokenize(self, prompt, images=None, max_length=None, **kwargs):
        length = max(len(prompt), 1)
        input_ids = mx.arange(length, dtype=mx.int32)[None, :]
        attention_mask = mx.ones((1, length), dtype=mx.int32)
        return type("TokenizerOutput", (), {"input_ids": input_ids, "attention_mask": attention_mask})()


class _FakeKrea2TextEncoder:
    def get_prompt_embeds(self, input_ids, attention_mask=None):
        return mx.zeros((1, input_ids.shape[1], 8), dtype=mx.float32)


class _FakeKrea2Transformer:
    def __call__(self, hidden_states, timestep, context, attention_mask=None):
        return mx.ones_like(hidden_states)


class _FakeIdeogram4Tokenizer:
    def tokenize_one(self, prompt, max_length=None):
        return mx.arange(8, dtype=mx.int32)


class _FakeIdeogram4TextEncoder:
    def get_prompt_embeds(self, input_ids, attention_mask, position_ids):
        return mx.zeros((1, input_ids.shape[1], 53248), dtype=mx.float32)


class _FakeIdeogram4Transformer:
    config = type("Config", (), {"in_channels": 128})()

    def __call__(self, **kwargs):
        return mx.zeros_like(kwargs["x"])


class _FakeQwen3VLChatTokenizer:
    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors=None
    ):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


class _FakeBooguTextEncoder:
    def get_instruction_features(self, input_ids, attention_mask=None):
        return mx.zeros((1, input_ids.shape[1], 8), dtype=mx.float32)


class _FakeBooguTransformer:
    def __call__(self, hidden_states, timestep, instruction_hidden_states):
        return mx.zeros_like(hidden_states)


class _RecordingVAE:
    spatial_scale = 8

    def __init__(self):
        self.decode_calls: list[tuple[int, ...]] = []

    def decode(self, latent):
        self.decode_calls.append(latent.shape)
        if latent.ndim == 5:
            b, _, t, h, w = latent.shape
            return mx.zeros((b, 3, t, h * 8, w * 8))
        b, _, h, w = latent.shape
        return mx.zeros((b, 3, h * 8, w * 8))


class _RecordingDecodeFlux2VAE(Flux2VAE):
    def __init__(self):
        nn.Module.__init__(self)
        self.bn = SimpleNamespace(running_mean=mx.zeros((128,)), running_var=mx.ones((128,)), eps=0.0)
        self.decode_calls: list[tuple[int, ...]] = []

    def decode(self, latents):
        self.decode_calls.append(latents.shape)
        if latents.ndim == 5:
            b, _, t, h, w = latents.shape
            return mx.zeros((b, 3, t, h * 8, w * 8))
        b, _, h, w = latents.shape
        return mx.zeros((b, 3, h * 8, w * 8))


class _NonPackedModelFixtures:
    @staticmethod
    def krea2(vae: _RecordingVAE, tiling_config: TilingConfig | None) -> Krea2:
        model = Krea2.__new__(Krea2)
        model.model_config = ModelConfig.krea2()
        model.callbacks = CallbackRegistry()
        model.tokenizers = {"qwen3vl": _FakeKrea2Tokenizer()}
        model.transformer = _FakeKrea2Transformer()
        model.vae = vae
        model.text_encoder = _FakeKrea2TextEncoder()
        model.prompt_cache = {}
        model.tiling_config = tiling_config
        model.bits = None
        model.lora_paths = None
        model.lora_scales = None
        return model

    @staticmethod
    def ideogram4(vae: _RecordingVAE, tiling_config: TilingConfig | None) -> Ideogram4:
        model = Ideogram4.__new__(Ideogram4)
        model.model_config = ModelConfig.ideogram4_fp8()
        model.callbacks = CallbackRegistry()
        model.tokenizers = {"ideogram4": _FakeIdeogram4Tokenizer()}
        model.conditional_transformer = _FakeIdeogram4Transformer()
        model.unconditional_transformer = _FakeIdeogram4Transformer()
        model.vae = vae
        model.bits = None
        model.lora_paths = None
        model.lora_scales = None
        model.prompt_cache = {}
        model.text_encoder = _FakeIdeogram4TextEncoder()
        model.tiling_config = tiling_config
        return model

    @staticmethod
    def ideogram4_caption() -> dict:
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

    @staticmethod
    def boogu(vae: _RecordingVAE, tiling_config: TilingConfig | None) -> BooguImage:
        model = BooguImage.__new__(BooguImage)
        model.model_config = ModelConfig.boogu_image_turbo()
        model.callbacks = CallbackRegistry()
        model.tokenizers = {"qwen3vl": SimpleNamespace(tokenizer=_FakeQwen3VLChatTokenizer())}
        model.text_encoder = _FakeBooguTextEncoder()
        model.transformer = _FakeBooguTransformer()
        model.vae = vae
        model.bits = None
        model.lora_paths = None
        model.lora_scales = None
        model.tiling_config = tiling_config
        return model


# Bug: flux2_klein.py's `decoded = self.vae.decode_packed_latents(packed_latents)` dropped
# the model's tiling_config, so --vae-tiling/--low-ram never tiled the Klein VAE decode.
@pytest.mark.fast
def test_flux2_klein_decode_passes_tiling_config_to_vae():
    vae = _RecordingPackedVAE()
    tiling_config = TilingConfig(vae_decode_tile_size=512)
    model = _PackedModelFixtures.flux2_klein(vae, tiling_config)

    model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64)

    assert vae.received_tiling_configs[-1] is tiling_config


@pytest.mark.fast
def test_flux2_klein_decode_with_tiling_config_none_still_returns_image():
    vae = _RecordingPackedVAE()
    model = _PackedModelFixtures.flux2_klein(vae, None)

    model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64)

    assert vae.received_tiling_configs[-1] is None


# Bug: flux2_klein_edit.py's `decoded = self.vae.decode_packed_latents(packed_latents)` dropped
# the model's tiling_config the same way as the txt2img variant.
@pytest.mark.fast
def test_flux2_klein_edit_decode_passes_tiling_config_to_vae(tmp_path):
    reference_path = tmp_path / "reference.png"
    _PackedModelFixtures.write_reference_image(reference_path)
    vae = _RecordingPackedVAE()
    tiling_config = TilingConfig(vae_decode_tile_size=512)
    model = _PackedModelFixtures.flux2_klein_edit(vae, tiling_config)

    model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64, image_paths=[reference_path])

    assert vae.received_tiling_configs[-1] is tiling_config


@pytest.mark.fast
def test_flux2_klein_edit_decode_with_tiling_config_none_still_returns_image(tmp_path):
    reference_path = tmp_path / "reference.png"
    _PackedModelFixtures.write_reference_image(reference_path)
    vae = _RecordingPackedVAE()
    model = _PackedModelFixtures.flux2_klein_edit(vae, None)

    model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64, image_paths=[reference_path])

    assert vae.received_tiling_configs[-1] is None


# Bug: lens_image.py's `decoded = self.vae.decode_packed_latents(packed.astype(mx.float32))`
# dropped tiling_config too, so --vae-tiling had no effect on Lens's decode.
@pytest.mark.fast
def test_lens_image_decode_passes_tiling_config_to_vae():
    vae = _RecordingPackedVAE()
    tiling_config = TilingConfig(vae_decode_tile_size=512)
    model = _PackedModelFixtures.lens_image(vae, tiling_config)

    model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64)

    assert vae.received_tiling_configs[-1] is tiling_config


@pytest.mark.fast
def test_lens_image_decode_with_tiling_config_none_still_returns_image():
    vae = _RecordingPackedVAE()
    model = _PackedModelFixtures.lens_image(vae, None)

    model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64)

    assert vae.received_tiling_configs[-1] is None


# Guard: Lens's decode now reads self.tiling_config; without this init line a plain Lens
# run (no --vae-tiling / --low-ram) would raise AttributeError at the decode.
@pytest.mark.fast
def test_lens_image_init_sets_tiling_config_to_none_by_default():
    instance = LensImage.__new__(LensImage)
    with patch.object(PathResolution, "resolve", side_effect=RuntimeError("stop-before-weight-loading")):
        with pytest.raises(RuntimeError, match="stop-before-weight-loading"):
            LensImage.__init__(instance, model_config=ModelConfig.lens_turbo())

    assert instance.tiling_config is None


# Bug: krea2.py's `_decode_latents` called `self.vae.decode(latents)` directly instead of
# routing through VAEUtil, so --vae-tiling never tiled the Krea 2 decode.
@pytest.mark.fast
def test_krea2_decode_tiles_when_tiling_config_set():
    vae = _RecordingVAE()
    tiling_config = TilingConfig(vae_decode_tile_size=128)
    model = _NonPackedModelFixtures.krea2(vae, tiling_config)

    model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=256, width=256, scheduler="euler")

    assert len(vae.decode_calls) > 1
    assert all(shape[-2] < 32 and shape[-1] < 32 for shape in vae.decode_calls)


@pytest.mark.fast
def test_krea2_decode_single_pass_when_tiling_config_none():
    vae = _RecordingVAE()
    model = _NonPackedModelFixtures.krea2(vae, None)

    model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=256, width=256, scheduler="euler")

    assert vae.decode_calls == [(1, 16, 32, 32)]


# Bug: ideogram4.py's `_decode_latents` called `self.vae.decode(latents)` directly instead of
# routing through VAEUtil, so --vae-tiling never tiled the Ideogram 4 decode.
@pytest.mark.fast
def test_ideogram4_decode_tiles_when_tiling_config_set():
    vae = _RecordingVAE()
    tiling_config = TilingConfig(vae_decode_tile_size=128)
    model = _NonPackedModelFixtures.ideogram4(vae, tiling_config)

    model.generate_image(
        prompt=_NonPackedModelFixtures.ideogram4_caption(), seed=1, num_inference_steps=1, width=256, height=256
    )

    assert len(vae.decode_calls) > 1
    assert all(shape[-2] < 32 and shape[-1] < 32 for shape in vae.decode_calls)


@pytest.mark.fast
def test_ideogram4_decode_single_pass_when_tiling_config_none():
    vae = _RecordingVAE()
    model = _NonPackedModelFixtures.ideogram4(vae, None)

    model.generate_image(
        prompt=_NonPackedModelFixtures.ideogram4_caption(), seed=1, num_inference_steps=1, width=256, height=256
    )

    assert vae.decode_calls == [(1, 32, 32, 32)]


# Bug: boogu_image.py's `decoded = self.vae.decode(latents)` called the VAE directly instead
# of routing through VAEUtil, so --vae-tiling never tiled the Boogu decode.
@pytest.mark.fast
def test_boogu_decode_tiles_when_tiling_config_set():
    vae = _RecordingVAE()
    tiling_config = TilingConfig(vae_decode_tile_size=128)
    model = _NonPackedModelFixtures.boogu(vae, tiling_config)

    model.generate_image(seed=1, prompt="a cat", num_inference_steps=1, height=256, width=256)

    assert len(vae.decode_calls) > 1
    assert all(shape[-2] < 32 and shape[-1] < 32 for shape in vae.decode_calls)


@pytest.mark.fast
def test_boogu_decode_single_pass_when_tiling_config_none():
    vae = _RecordingVAE()
    model = _NonPackedModelFixtures.boogu(vae, None)

    model.generate_image(seed=1, prompt="a cat", num_inference_steps=1, height=256, width=256)

    assert vae.decode_calls == [(1, 16, 32, 32)]


# Guard: the three FLUX.2-VAE call sites rely on decode_packed_latents forwarding tiling_config
# to VAEUtil.decode; if it dropped the config, they would never tile.
@pytest.mark.fast
def test_flux2_vae_decode_packed_latents_tiles_when_tiling_config_set():
    vae = _RecordingDecodeFlux2VAE()

    vae.decode_packed_latents(mx.zeros((1, 128, 16, 16)), tiling_config=TilingConfig(vae_decode_tile_size=128))

    assert len(vae.decode_calls) > 1
    assert all(shape[-2] < 32 and shape[-1] < 32 for shape in vae.decode_calls)


@pytest.mark.fast
def test_flux2_vae_decode_packed_latents_single_pass_when_tiling_config_none():
    vae = _RecordingDecodeFlux2VAE()

    vae.decode_packed_latents(mx.zeros((1, 128, 16, 16)))

    assert vae.decode_calls == [(1, 32, 32, 32)]
