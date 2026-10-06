import json
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen21 import qwen_image21_initializer
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

pytestmark = pytest.mark.fast


class TinyText:
    @staticmethod
    def make(bits=None):
        model = SimpleNamespace(
            vae=nn.Sequential(nn.Linear(64, 64, bias=False)),
            transformer=nn.Linear(64, 64, bias=False),
            text_encoder=nn.Sequential(nn.Linear(64, 64, bias=False)),
            tokenizers={},
            bits=bits,
        )
        if bits:
            for component in Qwen21WeightDefinition.get_components():
                if not component.skip_quantization:
                    nn.quantize(getattr(model, component.name), bits=bits)
        return model

    @staticmethod
    def init_models(model):
        components = TinyText.make()
        for name in ("vae", "transformer", "text_encoder"):
            setattr(model, name, getattr(components, name))

    @staticmethod
    def use_tiny_components(monkeypatch):
        monkeypatch.setattr(Qwen21Initializer, "_init_models", TinyText.init_models)
        monkeypatch.setattr(Qwen21Initializer, "_init_tokenizers", lambda model, path: None)


class TinyEdit:
    CONFIG = dict(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=64,
        axes_dims_rope=(16, 24, 24),
        context_in_dim=64,
        in_channels=4,
        out_channels=4,
        mlp_ratio=2,
        eps=1e-6,
        causal_condition=True,
    )

    @staticmethod
    def make(bits=None):
        model = QwenImage21Edit.__new__(QwenImage21Edit)
        nn.Module.__init__(model)
        model.transformer = QwenImage21Transformer(TinyEdit.CONFIG)
        model.vae = nn.Linear(64, 64, bias=False)
        model.text_encoder = nn.Linear(64, 64, bias=False)
        model.tokenizers = {}
        model.bits = bits
        model._component_configs = {name: {} for name in ("vae", "text_encoder")}
        model._component_configs["transformer"] = TinyEdit.CONFIG
        model._checkpoint_path = "/missing-original-checkpoint"
        model.processor = SimpleNamespace(save_pretrained=lambda path: None)
        if bits:
            for component in QwenImage21WeightDefinition.get_components():
                if not component.skip_quantization:
                    nn.quantize(
                        getattr(model, component.name),
                        bits=bits,
                        class_predicate=QwenImage21WeightDefinition.quantization_predicate,
                    )
        return model

    @staticmethod
    def output(model):
        layout = QwenImage21Layout.create(mx.array([False, True, False]), [(1, 2, 2)] * 2, (16, 24, 24))
        hidden = mx.arange(32, dtype=mx.float32).reshape(1, 8, 4) / 32
        text = mx.arange(192, dtype=mx.float32).reshape(1, 3, 64) / 192
        return np.array(model.transformer(hidden, text, mx.array([0.6]), layout))

    @staticmethod
    def use_tiny_components(monkeypatch):
        monkeypatch.setattr(qwen_image21_initializer, "QwenImage21VAE", lambda config: nn.Linear(64, 64, bias=False))
        monkeypatch.setattr(
            qwen_image21_initializer, "QwenImage21TextEncoder", lambda config: nn.Linear(64, 64, bias=False)
        )
        monkeypatch.setattr(
            qwen_image21_initializer.TokenizerLoader,
            "load_all",
            lambda *args: {"qwen21": SimpleNamespace(tokenizer=SimpleNamespace(save_pretrained=lambda path: None))},
        )
        monkeypatch.setattr(
            qwen_image21_initializer,
            "QwenImage21Processor",
            lambda *args: SimpleNamespace(save_pretrained=lambda path: None),
        )


@pytest.mark.parametrize("bits", [None, 8])
def test_text_initializer_validates_and_reloads_compatible_exports(tmp_path, monkeypatch, bits):
    original = TinyText.make(bits)
    ModelSaver.save_model(original, bits, str(tmp_path), Qwen21WeightDefinition)
    TinyText.use_tiny_components(monkeypatch)

    restored = QwenImage21(model_path=str(tmp_path))

    assert restored.bits == bits
    for component in Qwen21WeightDefinition.get_components():
        expected = dict(tree_flatten(getattr(original, component.name).parameters()))
        actual = dict(tree_flatten(getattr(restored, component.name).parameters()))
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(np.array(actual[key]), np.array(expected[key]))


@pytest.mark.parametrize("bits", [None, 8])
def test_text_initializer_rejects_edit_export(tmp_path, monkeypatch, bits):
    TinyEdit.make(bits).save_model(str(tmp_path))
    TinyText.use_tiny_components(monkeypatch)

    with pytest.raises(ValueError, match="vae checkpoint mismatch"):
        QwenImage21(model_path=str(tmp_path))


@pytest.mark.parametrize("levels", [(4, 8, 8), (8, 4, 4), (None, 8, 4)])
def test_shared_loading_rejects_conflicting_component_quantization(tmp_path, levels):
    for component, bits in zip(Qwen21WeightDefinition.get_components(), levels, strict=True):
        source = TinyText.make(bits)
        ModelSaver._save_weights(str(tmp_path), bits, getattr(source, component.name), component.hf_subdir)

    with pytest.raises(ValueError, match="Conflicting component quantization levels"):
        Qwen21Initializer.load_components(TinyText.make(), tmp_path, Qwen21WeightDefinition, None, validate=True)


@pytest.mark.parametrize("levels", [(None, 8, 8), (8, 8, None)])
def test_shared_loading_allows_dense_and_matching_quantized_components(tmp_path, levels):
    for component, bits in zip(Qwen21WeightDefinition.get_components(), levels, strict=True):
        source = TinyText.make(bits)
        ModelSaver._save_weights(str(tmp_path), bits, getattr(source, component.name), component.hf_subdir)
    restored = TinyText.make()

    Qwen21Initializer.load_components(restored, tmp_path, Qwen21WeightDefinition, None, validate=True)

    assert restored.bits == 8


class _HeadedTextEncoder(nn.Module):
    # A text encoder with a generation head beside its body, like the edit model's lm_head.
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(1024, 1024, bias=False)
        self.lm_head = nn.Linear(1024, 4096, bias=False)


@pytest.mark.parametrize(("saved_bits", "quantize"), [(None, None), (None, 8), (8, None)])
def test_shared_loading_reads_each_weight_at_its_first_use(tmp_path, saved_bits, quantize):
    # Evaluating every component at load held all of them at once, so --low-ram could not lower
    # the peak (#832). Measured as memory, so a read through any path counts, not only mx.eval:
    # nothing is read at load, and the first use of the generation head reads the head alone.
    original = TinyEdit.make(saved_bits)
    original.text_encoder = _HeadedTextEncoder()
    if saved_bits:
        nn.quantize(
            original.text_encoder,
            bits=saved_bits,
            class_predicate=QwenImage21WeightDefinition.quantization_predicate,
        )
    x = mx.ones((1, 1024))
    expected = original.text_encoder.lm_head(x)
    mx.eval(original.parameters(), expected)
    ModelSaver.save_model(original, saved_bits, str(tmp_path), QwenImage21WeightDefinition)
    restored = TinyEdit.make()
    restored.text_encoder = _HeadedTextEncoder()
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()

    Qwen21Initializer.load_components(restored, tmp_path, QwenImage21WeightDefinition, quantize, validate=True)

    assert mx.get_active_memory() - before < 1e6
    head = restored.text_encoder.lm_head(x)
    mx.eval(head)
    # The dense head that -q quantizes is released when the GPU finishes, a moment after eval returns.
    mx.synchronize()
    read = mx.get_active_memory() - before
    head_bytes, body_bytes = (
        sum(value.nbytes for _, value in tree_flatten(layer.parameters()))
        for layer in (restored.text_encoder.lm_head, restored.text_encoder.body)
    )
    assert head_bytes <= read < head_bytes + body_bytes
    assert mx.allclose(head, expected, atol=0.02).item()


@pytest.mark.parametrize("bits", [None, 8])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_native_hf_transformer_without_generated_buffers_loads(tmp_path, bits, dtype):
    class TransformerOnly:
        get_components = staticmethod(
            lambda: [replace(QwenImage21WeightDefinition.get_components()[1], precision=dtype)]
        )
        quantization_predicate = QwenImage21WeightDefinition.quantization_predicate

    original = TinyEdit.make()
    original.transformer.set_dtype(dtype)
    weights = {
        key.replace("modulation.layers.1.", "modulation.1."): value
        for key, value in tree_flatten(original.transformer.parameters())
        if not key.startswith("pos_embed.") and key != "time_text_embed.time_proj.freqs"
    }
    path = tmp_path / "transformer"
    path.mkdir()
    mx.save_safetensors(str(path / "diffusion_pytorch_model.safetensors"), weights)
    if bits:
        nn.quantize(original.transformer, bits=bits, class_predicate=TransformerOnly.quantization_predicate)
    restored = TinyEdit.make()

    Qwen21Initializer.load_components(restored, tmp_path, TransformerOnly, bits, validate=True)

    np.testing.assert_array_equal(TinyEdit.output(restored), TinyEdit.output(original))


@pytest.mark.parametrize("bits", [None, 8])
def test_legacy_edit_export_loads_with_same_outputs(tmp_path, bits):
    original = TinyEdit.make(bits)
    expected = TinyEdit.output(original)
    ModelSaver.save_model(original, bits, str(tmp_path), QwenImage21WeightDefinition)
    path = tmp_path / "transformer" / "0.safetensors"
    weights, metadata = mx.load(str(path), return_metadata=True)
    legacy = {key.replace("modulation.layers.1.", "modulation.1."): value for key, value in weights.items()}
    mx.save_safetensors(str(path.with_suffix(".tmp.safetensors")), legacy, metadata)
    path.with_suffix(".tmp.safetensors").replace(path)

    restored = TinyEdit.make()
    Qwen21Initializer.load_components(restored, tmp_path, QwenImage21WeightDefinition, None, validate=True)

    assert restored.bits == bits
    np.testing.assert_array_equal(TinyEdit.output(restored), expected)
    assert isinstance(restored.transformer.transformer_blocks[0].attn.to_q, nn.QuantizedLinear) is bool(bits)
    assert isinstance(restored.transformer.modulation.layers[1], nn.Linear)


@pytest.mark.parametrize("bits", [None, 8])
def test_old_text_export_keeps_packed_modulation_and_drops_frequency_buffer(tmp_path, bits):
    class TransformerOnly:
        get_components = staticmethod(lambda: [ComponentDefinition(name="transformer", hf_subdir="transformer")])
        get_tokenizers = staticmethod(lambda: [])
        quantization_predicate = staticmethod(
            lambda path, module: (
                Qwen21WeightDefinition.quantization_predicate(path, module) and module.weight.shape[-1] % 64 == 0
            )
        )

    model = SimpleNamespace(transformer=QwenImage21Transformer(TinyEdit.CONFIG))
    if bits:
        nn.quantize(model.transformer, bits=bits, class_predicate=TransformerOnly.quantization_predicate)
    ModelSaver.save_model(model, bits, str(tmp_path), TransformerOnly)
    path = tmp_path / "transformer" / "0.safetensors"
    weights, metadata = mx.load(str(path), return_metadata=True)
    # Exercise complete legacy paths, including packed weight/scales/biases.
    legacy = {key.replace("modulation.layers.1.", "modulation.1."): value for key, value in weights.items()}
    legacy["time_text_embed.time_proj.freqs"] = mx.arange(128)
    for table in ("cos_tables", "sin_tables"):
        for axis in range(3):
            legacy[f"pos_embed.{table}.{axis}"] = mx.zeros((4096, 8))
    mx.save_safetensors(str(path.with_suffix(".tmp.safetensors")), legacy, metadata)
    path.with_suffix(".tmp.safetensors").replace(path)
    restored = SimpleNamespace(transformer=QwenImage21Transformer(TinyEdit.CONFIG))

    Qwen21Initializer.load_components(restored, tmp_path, TransformerOnly, None, validate=True)

    for key, value in tree_flatten(model.transformer.parameters()):
        np.testing.assert_array_equal(
            np.array(dict(tree_flatten(restored.transformer.parameters()))[key]), np.array(value)
        )


@pytest.mark.parametrize("bits", [None, 8])
@pytest.mark.parametrize("bake_lora", [False, True])
def test_edit_lora_save_reload_preserves_adapter_once(tmp_path, monkeypatch, bits, bake_lora):
    model = TinyEdit.make(bits)
    before = TinyEdit.output(model)
    base_path = tmp_path / "base"
    model.save_model(str(base_path))
    adapter = tmp_path / "adapter.safetensors"
    matrices = {}
    for source, target in [
        ("transformer_blocks.0.attn.to_q", model.transformer.transformer_blocks[0].attn.to_q),
        ("modulation.1", model.transformer.modulation.layers[1]),
    ]:
        width = 64
        height = target.weight.shape[0]
        matrices[f"diffusion_model.{source}.lora_A.weight"] = mx.ones((2, width)) * 0.05
        matrices[f"diffusion_model.{source}.lora_B.weight"] = mx.arange(height * 2).reshape(height, 2) * 0.0005
    mx.save_safetensors(str(adapter), matrices)
    TinyEdit.use_tiny_components(monkeypatch)
    model = QwenImage21Edit(
        model_path=str(base_path), lora_paths=[str(adapter)], lora_scales=[0.7], bake_lora=bake_lora
    )
    adapted = TinyEdit.output(model)
    assert not np.allclose(adapted, before, atol=1e-5)
    assert isinstance(model.transformer.modulation.layers[1], LoRALinear) is (not bake_lora)
    saved_path = tmp_path / "export"

    model.save_model(str(saved_path))
    after_save = TinyEdit.output(model)
    restored = QwenImage21Edit(model_path=str(saved_path))

    np.testing.assert_array_equal(TinyEdit.output(restored), after_save)
    if bits is None:
        np.testing.assert_allclose(after_save, adapted, atol=1e-5, rtol=1e-5)
    assert not np.allclose(after_save, before, atol=1e-5)
    assert restored.lora_paths == []
    assert restored.lora_scales == []
    assert not any("lora_A" in key or "lora_B" in key for key, _ in tree_flatten(restored.parameters()))
    assert json.loads((saved_path / "transformer" / "config.json").read_text())["num_layers"] == 1


def test_edit_rejects_text_only_export_with_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="text-only exports do not contain the visual encoder"):
        QwenImage21Edit(model_path=str(tmp_path))


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape"])
def test_shared_loading_retains_edit_checkpoint_validation(tmp_path, problem):
    original = TinyEdit.make()
    ModelSaver.save_model(original, None, str(tmp_path), QwenImage21WeightDefinition)
    path = tmp_path / "text_encoder" / "0.safetensors"
    weights, metadata = mx.load(str(path), return_metadata=True)
    if problem == "missing":
        weights = {}
    elif problem == "unexpected":
        weights["surprise"] = mx.ones(1)
    else:
        weights["weight"] = mx.ones((1, 1))
    mx.save_safetensors(str(path.with_suffix(".tmp.safetensors")), weights, metadata)
    path.with_suffix(".tmp.safetensors").replace(path)
    with pytest.raises(ValueError, match="text_encoder checkpoint mismatch"):
        Qwen21Initializer.load_components(TinyEdit.make(), tmp_path, QwenImage21WeightDefinition, None, validate=True)


@pytest.mark.parametrize("prequantized", [False, True])
def test_text_encoder_retains_checkpoint_quantization_despite_on_load_skip(tmp_path, prequantized):
    class TextEncoderOnly:
        get_components = staticmethod(lambda: [Qwen21WeightDefinition.get_components()[2]])
        get_tokenizers = staticmethod(lambda: [])
        quantization_predicate = Qwen21WeightDefinition.quantization_predicate

    original = SimpleNamespace(text_encoder=nn.Sequential(nn.Linear(64, 64)))
    if prequantized:
        nn.quantize(original.text_encoder, bits=8)
    ModelSaver.save_model(original, 8, str(tmp_path), TextEncoderOnly)
    restored = SimpleNamespace(text_encoder=nn.Sequential(nn.Linear(64, 64)))

    Qwen21Initializer.load_components(restored, tmp_path, TextEncoderOnly, None, validate=True)

    assert restored.bits == 8
    assert isinstance(restored.text_encoder.layers[0], nn.QuantizedLinear) is prequantized
    inputs = mx.arange(64, dtype=mx.float32)[None] / 64
    np.testing.assert_array_equal(np.array(restored.text_encoder(inputs)), np.array(original.text_encoder(inputs)))
