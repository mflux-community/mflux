from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen21.model.qwen21_text_encoder.language_model import LanguageModel
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.model.qwen21_vae.vae import QwenImage21VAE
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

pytestmark = pytest.mark.fast


class TinyComponents:
    VAE = dict(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=[1, 2, 4, 8, 8],
        num_res_blocks=1,
        temperal_downsample=[False, True, True, True],
        is_residual=True,
        patch_size=None,
        in_channels=4,
        out_channels=4,
        latents_mean=[0.51, -0.31, 0.79, -1.25],
        latents_std=[1.31, 2.71, 3.14, 0.93],
    )

    @staticmethod
    def vae(edit, dtype=mx.float32):
        model = (QwenImage21VAE if edit else Qwen21VAE)(TinyComponents.VAE)
        model.set_dtype(dtype)
        return model

    @staticmethod
    def old_text_key(key):
        if key.endswith(".gamma"):
            return key.removesuffix(".gamma") + ".weight"
        if ".resample.1." in key:
            return key.replace(".resample.1.", ".conv.")
        parent, param = key.rsplit(".", 1)
        if parent.split(".")[-1] in {
            "conv_in",
            "conv_out",
            "conv1",
            "conv2",
            "conv_shortcut",
            "quant_conv",
            "post_quant_conv",
        }:
            return f"{parent}.conv.{param}"
        return key


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("format", ["hf", "legacy", "shared"])
def test_vae_loading_boundary_preserves_old_exports_and_native_weights(tmp_path, edit, dtype, format):
    class VAEOnly:
        get_components = staticmethod(
            lambda: [replace(QwenImage21WeightDefinition.get_components()[0], precision=dtype)]
        )
        quantization_predicate = QwenImage21WeightDefinition.quantization_predicate

    original = TinyComponents.vae(edit, dtype)
    supplied = dict(tree_flatten(original.parameters()))
    if format == "hf":
        (tmp_path / "vae").mkdir()
        tensors = {key: value.transpose(0, 3, 1, 2) if value.ndim == 4 else value for key, value in supplied.items()}
        mx.save_safetensors(str(tmp_path / "vae" / "diffusion_pytorch_model.safetensors"), tensors)
    else:
        ModelSaver._save_weights(str(tmp_path), None, original, "vae")
        if format == "legacy" and not edit:
            path = next((tmp_path / "vae").glob("*.safetensors"))
            tensors, metadata = mx.load(str(path), return_metadata=True)
            temporary = path.with_suffix(".tmp.safetensors")
            mx.save_safetensors(
                str(temporary), {TinyComponents.old_text_key(k): v for k, v in tensors.items()}, metadata
            )
            temporary.replace(path)
    restored = SimpleNamespace(vae=TinyComponents.vae(edit, dtype))
    Qwen21Initializer.load_components(restored, tmp_path, VAEOnly, None, validate=True)
    actual = dict(tree_flatten(restored.vae.parameters()))
    assert actual.keys() == supplied.keys()
    for key, expected in supplied.items():
        np.testing.assert_array_equal(np.array(actual[key].astype(mx.float32)), np.array(expected.astype(mx.float32)))
    latent = mx.linspace(-1, 1, 24).reshape(1, 4, 2, 3).astype(dtype)
    np.testing.assert_array_equal(
        np.array(restored.vae.decode(latent).astype(mx.float32)), np.array(original.decode(latent).astype(mx.float32))
    )


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
def test_shared_vae_keeps_color_shape_and_tiling_contracts(edit, tiled):
    model = TinyComponents.vae(edit)
    config = (
        TilingConfig(
            vae_encode_tile_size=32,
            vae_encode_tile_overlap=16,
            vae_decode_tile_size=32,
            vae_decode_overlap=1,
            vae_decode_tiles_per_dim=2,
        )
        if tiled
        else None
    )
    channels = 4 if edit else 3
    pixels = mx.linspace(-1, 1, channels * 32 * 48).reshape(1, channels, 32, 48)
    latent = VAEUtil.encode(model, pixels, config)
    assert latent.shape == ((1, 4, 1, 2, 3) if tiled else (1, 4, 2, 3))
    decoded = VAEUtil.decode(model, latent, config)
    assert decoded.shape == (1, channels, 32, 48)
    assert mx.all(mx.isfinite(decoded)).item()
    if edit:
        assert mx.max(mx.abs(decoded)).item() <= 1
    else:
        rgba = mx.concatenate([pixels, mx.ones_like(pixels[:, :1])], axis=1)
        np.testing.assert_array_equal(np.array(model.encode(pixels)), np.array(model.encode(rgba)))


@pytest.mark.parametrize("edit", [False, True])
def test_shared_vae_rejects_video_frames(edit):
    model = TinyComponents.vae(edit)
    with pytest.raises(ValueError, match="single-frame"):
        model.encode(mx.zeros((1, 4, 2, 32, 32)))
    with pytest.raises(ValueError, match="single-frame"):
        model.decode(mx.zeros((1, 4, 2, 2, 2)))


def test_vae_rejects_conflicting_legacy_and_shared_keys():
    with pytest.raises(ValueError, match="Duplicate VAE"):
        Qwen21Initializer._normalize_vae_weights(
            {"encoder.conv_in.conv.weight": mx.zeros((1,)), "encoder.conv_in.weight": mx.ones((1,))}
        )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_text_only_decoder_preserves_padding_without_vision_tower(dtype):
    model = Qwen21TextEncoder(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        head_dim=8,
        mrope_section=[2, 1, 1],
    )
    model.set_dtype(dtype)
    assert isinstance(model, LanguageModel)
    assert not hasattr(model, "visual")
    ids = mx.array([[1, 2, 3, 4]])
    masked = model(ids, mx.array([[1, 0, 1, 1]]))
    changed = model(mx.array([[1, 17, 3, 4]]), mx.array([[1, 0, 1, 1]]))
    np.testing.assert_array_equal(
        np.array(masked[:, 2:].astype(mx.float32)), np.array(changed[:, 2:].astype(mx.float32))
    )
    assert masked.shape == (1, 4, 16)


def test_vae_variants_share_core_without_sharing_mutable_parameters():
    text, edit = TinyComponents.vae(False), TinyComponents.vae(True)
    assert type(text.encoder) is type(edit.encoder)
    assert type(text.decoder) is type(edit.decoder)
    assert type(text.decoder.mid_block) is type(edit.decoder.mid_block)
    before = np.array(edit.encoder.conv_in.weight)
    text.encoder.conv_in.weight = mx.zeros_like(text.encoder.conv_in.weight)
    np.testing.assert_array_equal(np.array(edit.encoder.conv_in.weight), before)
