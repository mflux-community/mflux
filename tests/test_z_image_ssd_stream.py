import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.ssd_stream.streamer import StreamedBlock
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.variants import z_image as z_image_module
from mflux.models.z_image.variants.z_image import ZImage
from mflux.models.z_image.weights.z_image_weight_definition import (
    ZImageNoTextEncoderWeightDefinition,
    ZImageWeightDefinition,
)
from mflux.models.z_image.z_image_initializer import ZImageInitializer

pytestmark = pytest.mark.fast

GROUPS = ("noise_refiner", "context_refiner", "layers")


class ZImageStreamSupport:
    @staticmethod
    def save_component(folder, module, groups):
        folder.mkdir()
        tensors = {f"{group}.{i}.{name}": v for group in groups for i, block in enumerate(getattr(module, group)) for name, v in tree_flatten(block.parameters())}  # fmt: off
        mx.save_safetensors(str(folder / "0.safetensors"), tensors)
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(tensors, "0.safetensors")}))  # fmt: off

    @staticmethod
    def saved_tiny_transformer(tmp_path):
        transformer = ZImageTransformer(
            dim=16, n_layers=1, n_heads=2, in_channels=4, cap_feat_dim=8, axes_dims=[2, 2, 4], axes_lens=[64, 32, 32]
        )
        mx.eval(transformer.parameters())
        ZImageStreamSupport.save_component(tmp_path / "transformer", transformer, GROUPS)
        return transformer


def test_the_streamed_transformer_gives_the_normal_output_for_two_steps_in_a_row(tmp_path):
    mx.random.seed(0)
    transformer = ZImageTransformer(
        dim=16, n_layers=3, n_heads=2, in_channels=4, cap_feat_dim=8, axes_dims=[2, 2, 4], axes_lens=[64, 32, 32]
    )
    mx.eval(transformer.parameters())
    x = mx.random.normal((4, 1, 8, 8))
    cap = mx.random.normal((5, 8))
    sigmas = mx.array([1.0, 0.5, 0.0])
    expected = [np.array(transformer(x, step, sigmas, cap)) for step in (0, 1)]

    ZImageStreamSupport.save_component(tmp_path / "transformer", transformer, GROUPS)
    model = SimpleNamespace(small_text_encoder=True, transformer=transformer)
    ZImageInitializer._apply_ssd_stream(model, str(tmp_path), ZImageNoTextEncoderWeightDefinition)

    for group in GROUPS:
        assert all(isinstance(block, StreamedBlock) for block in getattr(transformer, group))
    for step in (0, 1):
        np.testing.assert_array_equal(np.array(transformer(x, step, sigmas, cap)), expected[step])


def test_the_streamed_text_encoder_gives_the_normal_output(tmp_path):
    mx.random.seed(1)
    encoder = TextEncoder(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        head_dim=8,
    )
    mx.eval(encoder.parameters())
    ids = mx.array([[1, 5, 7, 2, 9]])
    expected = np.array(encoder(ids).astype(mx.float32))

    ZImageStreamSupport.save_component(tmp_path / "text_encoder", encoder, ("layers",))
    transformer = ZImageStreamSupport.saved_tiny_transformer(tmp_path)
    model = SimpleNamespace(small_text_encoder=False, text_encoder=encoder, transformer=transformer)
    ZImageInitializer._apply_ssd_stream(model, str(tmp_path), ZImageWeightDefinition)

    assert all(isinstance(layer, StreamedBlock) for layer in encoder.layers)
    for _ in range(2):
        np.testing.assert_array_equal(np.array(encoder(ids).astype(mx.float32)), expected)


def test_the_small_text_encoder_is_not_streamed(tmp_path):
    transformer = ZImageStreamSupport.saved_tiny_transformer(tmp_path)
    small = SimpleNamespace(layers=["untouched"])
    model = SimpleNamespace(small_text_encoder=True, text_encoder=small, transformer=transformer)
    ZImageInitializer._apply_ssd_stream(model, str(tmp_path), ZImageNoTextEncoderWeightDefinition)
    assert small.layers == ["untouched"]


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [
        ({"quantize": 8}, "-q/--quantize"),
        ({"lora_paths": ["a.safetensors"], "bake_lora": True}, "--bake-lora"),
        ({"compute_precision": mx.float16}, "--compute-precision"),
    ],
)
def test_ssd_stream_rejects_incompatible_options_before_any_weight_is_read(kwargs, flag):
    args = {"quantize": None, "lora_paths": None, "bake_lora": True, "compute_precision": None} | kwargs
    with pytest.raises(ValueError, match=flag):
        ZImageInitializer._check_ssd_stream(**args)
    # init() checks first, so a nonexistent model path is never touched.
    with pytest.raises(ValueError, match=flag):
        ZImageInitializer.init(SimpleNamespace(), model_config=None, model_path="/nonexistent", ssd_stream=True, **args)


def test_ssd_stream_accepts_a_runtime_lora_and_no_options():
    ZImageInitializer._check_ssd_stream(None, None, True, None)
    ZImageInitializer._check_ssd_stream(None, ["a.safetensors"], False, None)


def test_save_model_refuses_a_streamed_model(tmp_path):
    model = SimpleNamespace(small_text_encoder=False, ssd_stream=True)
    with pytest.raises(ValueError, match="cannot be saved"):
        ZImage.save_model(model, str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("ssd_stream", [True, False])
def test_predict_is_compiled_unless_the_model_streams(monkeypatch, ssd_stream):
    compiled = []
    monkeypatch.setattr(z_image_module.mx, "compile", lambda fn: compiled.append(fn) or fn)
    monkeypatch.setattr(z_image_module.AppleSiliconUtil, "is_m1_or_m2", staticmethod(lambda: False))
    ZImage._predict(SimpleNamespace(), compile_predict=not ssd_stream)
    assert bool(compiled) is (not ssd_stream)
