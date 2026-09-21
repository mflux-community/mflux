import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.cli.defaults.defaults import model_inference_steps
from mflux.models.common.cli.save import MODEL_CLASSES
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.cli import qwen21_edit_generate as cli
from mflux.models.qwen21.cli.qwen21_edit_generate import build_parser
from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
from mflux.models.qwen21.reference.model.qwen_image21_vae.blocks import DownBlock
from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_image import GeneratedImage

pytestmark = pytest.mark.fast


class TestQwenImage21:
    @staticmethod
    def small_transformer():
        return QwenImage21Transformer(
            dict(
                num_layers=2,
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

    @pytest.mark.parametrize("name", ["qwen-2.1", "qwen-image-21", "qwen-image-2.1", "Qwen/Qwen-Image-2.1"])
    def test_registry_and_defaults(self, name):
        assert ModelConfig.from_name(name).model_name == "Qwen/Qwen-Image-2.1"
        assert model_inference_steps(name) == 40
        assert MODEL_CLASSES["qwen-image-2.1"].__name__ == "QwenImage21"

    def test_cli_default_and_cache_override(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["qwen21", "--prompt", "test", "--no-use-kv-cache"])
        args = build_parser().parse_args()
        assert args.steps == 40
        assert args.width is None and args.height is None
        assert args.use_kv_cache is False
        assert args.output_resolution == 1024
        assert ConfigResolution.resolve_restricted(args.model, "qwen-image-2.1") is ModelConfig.from_name("qwen-2.1")

    def test_cli_rejects_foreign_models(self):
        with pytest.raises(ModelConfigError, match="only accepts"):
            ConfigResolution.resolve_restricted("qwen", "qwen-image-2.1")

    @pytest.mark.parametrize("option,value", [("--width", "31"), ("--steps", "1"), ("--output-resolution", "17")])
    def test_invalid_cli_inputs_fail_before_loading(self, option, value, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["qwen21", "--prompt", "test", option, value])
        monkeypatch.setattr(cli, "QwenImage21Edit", lambda **kwargs: pytest.fail("Invalid input reached model loading"))
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2

    def test_metadata_restores_and_explicit_flags_override(self, tmp_path, monkeypatch):
        path = tmp_path / "meta.json"
        path.write_text(
            json.dumps(
                dict(
                    model="Qwen/Qwen-Image-2.1",
                    prompt="test",
                    seed=42,
                    use_kv_cache=False,
                    output_resolution=512,
                    width=64,
                    height=32,
                )
            )
        )
        monkeypatch.setattr(sys, "argv", ["qwen21", "--config-from-metadata", str(path)])
        args = build_parser().parse_args()
        assert (args.width, args.height, args.use_kv_cache, args.output_resolution) == (64, 32, False, 512)
        monkeypatch.setattr(
            sys,
            "argv",
            ["qwen21", "--config-from-metadata", str(path), "--use-kv-cache", "--output-resolution", "1024"],
        )
        args = build_parser().parse_args()
        assert args.use_kv_cache and args.output_resolution == 1024

    def test_latent_layout_roundtrip(self):
        x = mx.arange(64 * 2 * 4).reshape(1, 64, 1, 2, 4)
        packed = QwenImage21LatentCreator.pack_latents(x)
        assert packed.shape == (1, 8, 64)
        np.testing.assert_array_equal(np.array(QwenImage21LatentCreator.unpack_latents(packed, 32, 64)), np.array(x))

    @pytest.mark.parametrize(
        "width,height,steps,count", [(31, 32, 40, 0), (32, 48, 40, 0), (32, 32, 1, 0), (32, 32, 40, 11)]
    )
    def test_reject_invalid_inputs(self, width, height, steps, count):
        with pytest.raises(ValueError):
            QwenImage21LatentCreator.validate(width, height, steps, count)

    def test_adjacent_images_remain_independent_blocks(self):
        layout = QwenImage21Layout.create(mx.array([False, True, True, False]), [(1, 2, 2)] * 3, (4, 6, 6))
        assert layout.segments == [(0, 1, True), (1, 5, False), (5, 9, False), (9, 10, True)]
        assert layout.prefix_length == 10
        with pytest.raises(ValueError, match="slots"):
            QwenImage21Layout.create(mx.array([True]), [(1, 4, 4), (1, 2, 2)], (4, 6, 6))

    @pytest.mark.parametrize("side", [128, 256])
    def test_first_frame_downsampling_preserves_large_images(self, side):
        block = DownBlock(96, 192, num_blocks=0, down=True, temporal=True)
        result = np.array(block._shortcut(mx.ones((1, side, side, 96))))
        # The prepended zero frame occupies even channels; averaging the real
        # frame's constant 2x2 patches must leave the odd channels equal to one.
        np.testing.assert_array_equal(result[..., ::2], 0)
        np.testing.assert_array_equal(result[..., 1::2], 1)

    def test_cached_matches_uncached_across_timesteps(self):
        mx.random.seed(42)
        model = self.small_transformer()
        layout = QwenImage21Layout.create(mx.array([False, True, True, False]), [(1, 2, 2)] * 3, (4, 6, 6))
        hidden, text = mx.random.normal((1, 12, 4)), mx.random.normal((1, 4, 32))
        cache = []
        for t in [0.9, 0.6, 0.2]:
            actual = model(hidden, text, mx.array([t]), layout, cache)
            expected = model(hidden, text, mx.array([t]), layout)
            np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-4, atol=1e-4)
            hidden = hidden.at[:, -4:].add(actual * 0.05)
        assert len(cache) == 2
        assert cache[0][0].shape[2] == layout.prefix_length

    def test_padding_is_not_visible_to_target(self):
        mx.random.seed(12)
        model = self.small_transformer()
        layout = QwenImage21Layout.create(mx.array([False, False, False]), [(1, 2, 2)], (4, 6, 6))
        hidden, text = mx.random.normal((1, 4, 4)), mx.random.normal((1, 3, 32))
        mask = mx.array([[True, True, False]])
        expected = model(hidden, text, mx.array([0.5]), layout, encoder_hidden_states_mask=mask)
        text[:, 2] = 100
        actual = model(hidden, text, mx.array([0.5]), layout, encoder_hidden_states_mask=mask)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)

    def test_rgba_png_and_metadata_roundtrip(self, tmp_path):
        pixels = np.zeros((4, 8, 4), dtype=np.uint8)
        pixels[..., 0] = 128
        pixels[..., 3] = np.arange(8, dtype=np.uint8) * 30
        image = GeneratedImage(
            image=Image.fromarray(pixels),
            model_config=ModelConfig.qwen_image_21(),
            seed=42,
            prompt="transparent sticker",
            steps=40,
            guidance=1.0,
            precision=mx.bfloat16,
            quantization=8,
            generation_time=1,
            width=8,
            height=4,
            generation_parameters={"use_kv_cache": False, "output_resolution": 512},
        )
        path = tmp_path / "rgba.png"
        image.save(path, export_json_metadata=True)
        saved = Image.open(path)
        assert saved.mode == "RGBA"
        np.testing.assert_array_equal(np.array(saved), pixels)
        assert image._get_metadata()["use_kv_cache"] is False
        assert image.get_right_half()._get_metadata()["output_resolution"] == 512

    def test_weight_transforms_and_quantization_exclusions(self):
        x = mx.zeros((8, 4, 3, 3))
        assert QwenImage21WeightDefinition.vae_weight("conv.weight", x).shape == (8, 3, 3, 4)
        assert (
            QwenImage21WeightDefinition.text_key("model.language_model.layers.0.weight")
            == "language_model.layers.0.weight"
        )
        assert QwenImage21WeightDefinition.text_key("lm_head.weight") is None
        module = SimpleNamespace(to_quantized=lambda: None, weight=mx.zeros((64, 64)))
        assert QwenImage21WeightDefinition.quantization_predicate("transformer_blocks.0.attn.to_q", module)
        assert not QwenImage21WeightDefinition.quantization_predicate("modulation.1", module)
