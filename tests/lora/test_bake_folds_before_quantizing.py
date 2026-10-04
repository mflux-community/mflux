import gc
import weakref
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.dense_weight import dense_weight
from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.z_image.z_image_initializer import ZImageInitializer

OUT, IN, GROUP = 128, 256, 64


def _dense_weight(seed: int = 0) -> mx.array:
    # bfloat16, as a checkpoint loads.
    mx.random.seed(seed)
    return (mx.random.normal((OUT, IN)) * 0.04).astype(mx.bfloat16)


def _quantized(weight: mx.array, bits: int) -> nn.QuantizedLinear:
    linear = nn.Linear(IN, OUT, bias=False)
    linear.weight = weight
    return linear.to_quantized(group_size=GROUP, bits=bits)


def _small_adapter(base: nn.Module, seed: int = 1) -> LoRALinear:
    # The delta sits near a tenth of a q8 step, the size at which folding onto the
    # quantized grid rounds most of it away (#814).
    mx.random.seed(seed)
    lora = LoRALinear.from_linear(base, r=8)
    lora.lora_A = mx.random.normal((IN, 8)) * 0.005
    lora.lora_B = mx.random.normal((8, OUT)) * 0.005
    return lora


def _delta(lora: LoRALinear) -> mx.array:
    return (lora.scale * mx.transpose(mx.matmul(lora.lora_A, lora.lora_B))).astype(mx.float32)


def _codes(weight: mx.array, bits: int) -> mx.array:
    return mx.quantize(weight, group_size=GROUP, bits=bits)[0]


def _strength(baked: nn.Module, base: nn.Module, lora: LoRALinear) -> float:
    # Projection of the change the bake made to the weight onto the adapter's delta. 1.0 is the whole
    # adapter. Taken on the weights, so no matmul rounding enters it.
    delta = _delta(lora).reshape(-1)
    change = (dense_weight(baked) - dense_weight(base)).astype(mx.float32).reshape(-1)
    return float(mx.sum(change * delta) / mx.sum(delta * delta))


@pytest.mark.parametrize("bits", [4, 8])
def test_fold_equals_quantizing_the_merged_weight(bits):
    weight = _dense_weight()
    base = _quantized(weight, bits)
    lora = _small_adapter(base)

    folded = LoRASaver._fold_before_quantizing(base, [lora], weight)

    assert isinstance(folded, nn.QuantizedLinear)
    assert (folded.bits, folded.group_size) == (bits, GROUP)
    assert mx.array_equal(folded.weight, _codes(weight.astype(mx.float32) + _delta(lora), bits))
    # Float32 scales would promote every activation through the layer.
    assert folded.scales.dtype == base.scales.dtype == mx.bfloat16
    assert folded.biases.dtype == base.biases.dtype


def test_fold_keeps_the_whole_adapter_where_the_grid_fold_loses_part():
    weight = _dense_weight()
    base = _quantized(weight, 8)
    lora = _small_adapter(base)

    folded = LoRASaver._fold_before_quantizing(base, [lora], weight)
    on_grid = LoRASaver._bake_lora_into_linear(base, lora)

    assert _strength(folded, base, lora) == pytest.approx(1.0, abs=0.1)
    assert _strength(on_grid, base, lora) < 0.7


def test_fold_multiplies_low_precision_factors_in_float32():
    # A float16 file: the product of these factors sits in float16's subnormal range, where a
    # float16 matmul keeps only a few bits of it. The weight's q8 step is about as small.
    mx.random.seed(5)
    weight = (mx.random.normal((OUT, IN)) * 1e-5).astype(mx.bfloat16)
    base = _quantized(weight, 8)
    lora = LoRALinear.from_linear(base, r=8)
    lora.lora_A = (mx.random.normal((IN, 8)) * 3e-4).astype(mx.float16)
    lora.lora_B = (mx.random.normal((8, OUT)) * 3e-4).astype(mx.float16)
    exact = mx.transpose(mx.matmul(lora.lora_A.astype(mx.float32), lora.lora_B.astype(mx.float32)))

    folded = LoRASaver._fold_before_quantizing(base, [lora], weight)

    assert mx.array_equal(folded.weight, _codes(weight.astype(mx.float32) + lora.scale * exact, 8))


def test_fold_sums_stacked_adapters():
    weight = _dense_weight()
    base = _quantized(weight, 8)
    first, second = _small_adapter(base, seed=1), _small_adapter(base, seed=3)

    folded = LoRASaver._fold_before_quantizing(base, [first, second], weight)

    assert mx.array_equal(folded.weight, _codes(weight.astype(mx.float32) + _delta(first) + _delta(second), 8))


def test_a_stored_weight_the_layer_was_not_quantized_from_is_ignored():
    weight = _dense_weight()
    base = _quantized(weight, 8)
    lora = _small_adapter(base)

    assert LoRASaver._fold_before_quantizing(base, [lora], _dense_weight(seed=7)) is None
    assert LoRASaver._fold_before_quantizing(base, [lora], None) is None


def test_a_rescaled_stored_weight_is_ignored():
    # A rescale keeps the codes and changes only the scales and biases.
    weight = _dense_weight()
    base = _quantized(weight, 8)
    lora = _small_adapter(base)

    for factor in (2.0, 0.5, -1.0):
        assert LoRASaver._fold_before_quantizing(base, [lora], weight * factor) is None


@pytest.mark.parametrize("with_dora", [False, True])
def test_fold_applies_a_lokr_adapter_to_the_dense_weight(with_dora):
    weight = _dense_weight()
    base = _quantized(weight, 8)
    mx.random.seed(4)
    dora_scale = mx.linalg.norm(weight.astype(mx.float32), axis=1) * 1.01 if with_dora else None
    lokr = LoKrLinear.from_linear(
        base,
        lokr_w1=mx.random.normal((8, 16)) * 0.05,
        lokr_w2=mx.random.normal((16, 16)) * 0.005,
        dora_scale=dora_scale,
        scale=0.5,
    )

    folded = LoRASaver._fold_before_quantizing(base, [lokr], weight)

    merged = weight.astype(mx.float32)
    merged = merged + 0.5 * lokr.delta_weight(base_weight=merged)
    assert mx.array_equal(folded.weight, _codes(merged, 8))
    assert folded.scales.dtype == mx.bfloat16


def test_bake_and_strip_folds_a_lokr_layer_before_quantizing():
    weight = _dense_weight()
    base = _quantized(weight, 4)
    mx.random.seed(4)
    lokr = LoKrLinear.from_linear(
        base, lokr_w1=mx.random.normal((8, 16)) * 0.05, lokr_w2=mx.random.normal((16, 16)) * 0.005, scale=0.5
    )
    expected = _codes(weight.astype(mx.float32) + 0.5 * lokr.delta_weight(), 4)
    transformer = _Transformer([lokr])

    LoRASaver.bake_and_strip_lora(transformer, dense_weights={"blocks": [{"proj": {"weight": weight}}]})

    # The fold onto the grid would have re-quantized this q4 layer at q8.
    assert transformer.blocks[0].proj.bits == 4
    assert mx.array_equal(transformer.blocks[0].proj.weight, expected)


def test_a_dense_base_is_left_to_the_dense_fold():
    weight = _dense_weight()
    dense = nn.Linear(IN, OUT, bias=False)
    dense.weight = weight
    lora = _small_adapter(dense)

    assert LoRASaver._fold_before_quantizing(dense, [lora], weight) is None


class _Block(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.proj = layer


class _Transformer(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.blocks = [_Block(layer) for layer in layers]


def test_bake_and_strip_folds_the_layers_found_in_the_dense_weights(capsys):
    stored, missing = _dense_weight(seed=0), _dense_weight(seed=5)
    stored_base, missing_base = _quantized(stored, 4), _quantized(missing, 4)
    stored_lora, missing_lora = _small_adapter(stored_base), _small_adapter(missing_base)
    fused = FusedLoRALinear(base_linear=_quantized(stored, 4), loras=[_small_adapter(stored_base, seed=3)])
    transformer = _Transformer([stored_lora, missing_lora, fused])
    dense_weights = {"blocks": [{"proj": {"weight": stored}}, {}, {"proj": {"weight": stored}}]}

    LoRASaver.bake_and_strip_lora(transformer, dense_weights=dense_weights)

    # Found in the tree: folded before quantizing, at the precision the layer had.
    assert transformer.blocks[0].proj.bits == 4
    assert mx.array_equal(transformer.blocks[0].proj.weight, _codes(stored.astype(mx.float32) + _delta(stored_lora), 4))
    assert transformer.blocks[2].proj.bits == 4
    # Not in the tree: folded onto the grid, with the q8 escape for a sub-8-bit layer.
    assert transformer.blocks[1].proj.bits == 8
    assert "Re-quantized 1 sub-8-bit layers at q8" in capsys.readouterr().out


def test_bake_and_strip_does_not_keep_the_dense_weights_alive():
    # On Krea 2 the 26 GB of dense weights stayed in memory after the load: the walker's
    # closures form a cycle, and they held the tree until a garbage collection.
    class _Tree(dict):
        pass

    weight = _dense_weight()
    transformer = _Transformer([_small_adapter(_quantized(weight, 8))])
    tree = _Tree(blocks=[{"proj": {"weight": weight}}])
    alive = weakref.ref(tree)

    gc.disable()
    try:
        LoRASaver.bake_and_strip_lora(transformer, dense_weights=tree)
        del tree
        assert alive() is None
    finally:
        gc.enable()


def test_a_failed_bake_does_not_keep_the_dense_weights_alive():
    class _Tree(dict):
        pass

    weight = _dense_weight()
    lora = _small_adapter(_quantized(weight, 8))
    lora.lora_B = mx.zeros((8, OUT // 2))  # does not fit the layer
    transformer = _Transformer([lora])
    tree = _Tree(blocks=[{"proj": {"weight": weight}}])
    alive = weakref.ref(tree)

    gc.disable()
    try:
        with pytest.raises((ValueError, RuntimeError)):
            LoRASaver.bake_and_strip_lora(transformer, dense_weights=tree)
        del tree
        assert alive() is None
    finally:
        gc.enable()


def test_bake_and_strip_without_dense_weights_is_unchanged():
    weight = _dense_weight()
    transformer = _Transformer([_small_adapter(_quantized(weight, 4))])
    reference = LoRASaver._bake_lora_into_linear(_quantized(weight, 4), _small_adapter(_quantized(weight, 4)))

    LoRASaver.bake_and_strip_lora(transformer)

    assert transformer.blocks[0].proj.bits == 8
    assert mx.array_equal(transformer.blocks[0].proj.weight, reference.weight)


def test_loader_passes_the_dense_weights_to_the_bake(tmp_path):
    weight = _dense_weight()
    transformer = _Transformer([_quantized(weight, 8)])
    mx.random.seed(1)
    lora_a, lora_b = mx.random.normal((8, IN)) * 0.005, mx.random.normal((OUT, 8)) * 0.005
    adapter = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(adapter), {"blocks.0.proj.lora_A.weight": lora_a, "blocks.0.proj.lora_B.weight": lora_b})
    mapping = [
        LoRATarget(
            model_path="blocks.0.proj",
            possible_up_patterns=["blocks.0.proj.lora_B.weight"],
            possible_down_patterns=["blocks.0.proj.lora_A.weight"],
        )
    ]

    LoRALoader.load_and_apply_lora(
        lora_mapping=mapping,
        transformer=transformer,
        lora_paths=[str(adapter)],
        lora_scales=[0.5],
        dense_weights={"blocks": [{"proj": {"weight": weight}}]},
    )

    delta = 0.5 * mx.transpose(mx.matmul(mx.transpose(lora_a), mx.transpose(lora_b)))
    merged = weight.astype(mx.float32) + delta.astype(mx.float32)
    assert mx.array_equal(transformer.blocks[0].proj.weight, _codes(merged, 8))


def test_dense_component_is_only_offered_for_an_unquantized_checkpoint():
    tree = {"proj": {"weight": _dense_weight()}}

    dense = LoadedWeights(components={"transformer": tree}, meta_data=MetaData(quantization_level=None))
    quantized = LoadedWeights(components={"transformer": tree}, meta_data=MetaData(quantization_level=8))

    assert dense.dense_component("transformer") is tree
    assert dense.dense_component("text_encoder") is None
    assert quantized.dense_component("transformer") is None


@pytest.mark.parametrize(
    "initializer",
    [FluxInitializer, Flux2Initializer, QwenImageInitializer, Krea2Initializer, ZImageInitializer],
)
@pytest.mark.parametrize("stored_bits", [None, 8])
def test_initializers_hand_the_loaded_transformer_weights_to_the_loader(monkeypatch, initializer, stored_bits):
    tree = {"proj": {"weight": _dense_weight()}}
    weights = LoadedWeights(components={"transformer": tree}, meta_data=MetaData(quantization_level=stored_bits))
    seen = {}

    def _record(**kwargs):
        seen.update(kwargs)
        return [], []

    monkeypatch.setattr(LoRALoader, "load_and_apply_lora", staticmethod(_record))

    initializer._apply_lora(SimpleNamespace(transformer=nn.Module()), ["adapter"], [1.0], True, weights)

    assert seen["dense_weights"] is (tree if stored_bits is None else None)


class _Stop(Exception):
    pass


_INIT_HELPERS = (
    "_init_config",
    "_init_tokenizers",
    "_init_models",
    "_init_edit_models",
    "_apply_weights",
)


@pytest.mark.parametrize(
    ("initializer", "method"),
    [
        (FluxInitializer, "init"),
        (FluxInitializer, "init_concept"),
        (Flux2Initializer, "init"),
        (QwenImageInitializer, "init"),
        (QwenImageInitializer, "init_edit"),
        (Krea2Initializer, "init"),
        (ZImageInitializer, "init"),
    ],
)
def test_every_init_passes_the_loaded_weights_to_apply_lora(monkeypatch, initializer, method):
    loaded = LoadedWeights(components={"transformer": {}}, meta_data=MetaData(quantization_level=None))
    seen = []
    for name in _INIT_HELPERS:
        if hasattr(initializer, name):
            monkeypatch.setattr(initializer, name, staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(initializer, "_load_weights", staticmethod(lambda *args, **kwargs: loaded))

    def _apply_lora(*args, **kwargs):
        # Stops the init here: what follows (tokenizer wrapping in init_edit) needs real parts.
        seen.append(args)
        raise _Stop

    monkeypatch.setattr(initializer, "_apply_lora", staticmethod(_apply_lora))
    if method == "init_concept":
        import mflux.models.flux.variants.concept_attention.transformer_concept as concept

        monkeypatch.setattr(concept, "TransformerConcept", lambda **kwargs: None)
    model, config = SimpleNamespace(), SimpleNamespace(model_name="unused")

    with pytest.raises(_Stop):
        getattr(initializer, method)(model=model, model_config=config, quantize=8, lora_paths=["adapter"])

    assert len(seen) == 1
    assert seen[0][-1] is loaded
