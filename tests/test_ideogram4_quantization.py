import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.ideogram4.ideogram4_initializer import Ideogram4Initializer
from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition

IN_DIMS, OUT_DIMS = 256, 128


def _loaded_fp8_linear(bias: bool = True) -> Fp8Linear:
    """An Fp8Linear holding real fp8 codes, as it looks once a checkpoint is applied."""
    layer = Fp8Linear(IN_DIMS, OUT_DIMS, bias=bias)
    weight = mx.random.normal((OUT_DIMS, IN_DIMS)).astype(mx.bfloat16) * 0.05
    scale = mx.max(mx.abs(weight), axis=1) / 448.0
    layer.weight = mx.to_fp8(weight / scale[:, None])
    layer.weight_scale = scale.astype(mx.float32)
    if bias:
        layer.bias = mx.random.normal((OUT_DIMS,)).astype(mx.bfloat16)
    return layer


@pytest.mark.fast
@pytest.mark.parametrize("bits,tolerance", [(8, 0.02), (4, 0.12)])
def test_to_quantized_preserves_the_forward(bits: int, tolerance: float):
    # Quantizing must approximate the fp8 layer it replaces, tighter at 8 bits than at 4.
    mx.random.seed(0)
    layer = _loaded_fp8_linear()
    x = mx.random.normal((4, IN_DIMS)).astype(mx.bfloat16)
    expected = layer(x).astype(mx.float32)

    quantized = layer.to_quantized(group_size=64, bits=bits)

    assert isinstance(quantized, nn.QuantizedLinear)
    assert quantized.bits == bits
    assert quantized.group_size == 64
    actual = quantized(x).astype(mx.float32)
    relative_error = mx.mean(mx.abs(actual - expected)) / mx.mean(mx.abs(expected))
    assert float(relative_error) < tolerance


@pytest.mark.fast
def test_to_quantized_before_weights_are_loaded_returns_a_usable_skeleton():
    # WeightApplier quantizes the skeleton *before* update() for an already-quantized
    # checkpoint, so to_quantized has to work on a layer that holds nothing yet. Large
    # layers skip eager init, which is exactly the case that reaches this path.
    layer = Fp8Linear(4096, 2048, bias=False)
    assert layer.weight.shape != (2048, 4096), "expected a lazily initialised layer"

    quantized = layer.to_quantized(group_size=64, bits=4)

    assert isinstance(quantized, nn.QuantizedLinear)
    assert quantized.weight.shape == (2048, 4096 * 4 // 32)
    assert quantized.scales.shape == (2048, 4096 // 64)
    assert "bias" not in quantized


@pytest.mark.fast
def test_bias_is_carried_over_and_never_invented():
    mx.random.seed(0)
    with_bias = _loaded_fp8_linear(bias=True).to_quantized(group_size=64, bits=8)
    without_bias = _loaded_fp8_linear(bias=False).to_quantized(group_size=64, bits=8)

    assert "bias" in with_bias
    assert "bias" not in without_bias


@pytest.mark.fast
def test_nn_quantize_reaches_fp8_layers():
    # The regression this guards: without to_quantized, the default predicate
    # (hasattr(module, "to_quantized")) skipped every Ideogram 4 layer, so `mflux-save -q`
    # wrote an unquantized checkpoint stamped with a quantization level.
    mx.random.seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = _loaded_fp8_linear()
            self.o = _loaded_fp8_linear(bias=False)

    block = Block()
    assert Ideogram4WeightDefinition.quantization_predicate("qkv", block.qkv)

    nn.quantize(block, group_size=64, bits=8)

    assert isinstance(block.qkv, nn.QuantizedLinear)
    assert isinstance(block.o, nn.QuantizedLinear)


@pytest.mark.fast
def test_transformers_and_text_encoder_are_quantizable():
    # These three carry every weight-bearing layer in the model; skipping them made
    # `mflux-save -q` a no-op for everything except the VAE.
    components = {c.name: c for c in Ideogram4WeightDefinition.get_components()}

    for name in ("conditional_transformer", "unconditional_transformer", "text_encoder"):
        assert not components[name].skip_quantization, f"{name} must participate in quantization"


@pytest.mark.fast
@pytest.mark.parametrize("bits", [8, 4])
def test_rebuild_matches_the_stored_bit_width(bits: int):
    # A q4 save packs twice as many weights per uint32 as a q8 one. Rebuilding at a
    # hard-coded 8 bits would size the layer wrongly and update() would skip it.
    mx.random.seed(0)

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = Fp8Linear(IN_DIMS, OUT_DIMS, bias=True)

    source = nn.Linear(IN_DIMS, OUT_DIMS, bias=True)
    source.weight = mx.random.normal((OUT_DIMS, IN_DIMS)).astype(mx.bfloat16) * 0.05
    stored = source.to_quantized(group_size=64, bits=bits)
    tree = {"qkv": dict(stored.parameters())}

    holder = Holder()
    Ideogram4Initializer._rebuild_q8_folded_layers(holder, tree)

    assert isinstance(holder.qkv, nn.QuantizedLinear)
    assert holder.qkv.bits == bits
    assert holder.qkv.group_size == 64
    # strict=True is the real assertion: every stored tensor must find a home.
    holder.update(tree, strict=True)

    x = mx.random.normal((2, IN_DIMS)).astype(mx.bfloat16)
    assert mx.allclose(holder.qkv(x), stored(x), atol=1e-4, rtol=1e-3)
