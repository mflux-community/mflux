import mlx.core as mx
from mlx import nn


def is_fp8_linear(linear) -> bool:
    # fp8 layers (e.g. Ideogram 4's Fp8Linear) store raw uint8 codes in .weight plus a
    # per-row weight_scale. A float delta cannot be folded into the codes directly:
    # `delta.astype(uint8)` truncates the (small) LoRA delta to zero, silently destroying
    # the adapter while leaving the base intact.
    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.dtype == mx.uint8
        and hasattr(linear, "weight_scale")
        and not isinstance(linear, nn.QuantizedLinear)
    )


def dense_weight(linear) -> mx.array:
    """The base layer's weight as real numbers, whatever it is stored as.

    Adapter deltas are computed against, and folded into, the actual weight values, so
    quantized and fp8 bases have to be decoded first. DoRA in particular is a non-linear
    function of the base weight, and would otherwise take a norm over packed codes.
    """
    if isinstance(linear, nn.QuantizedLinear):
        return mx.dequantize(
            linear.weight,
            linear.scales,
            biases=linear.biases,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
        )
    if is_fp8_linear(linear):
        return mx.from_fp8(linear.weight, dtype=mx.float32) * linear.weight_scale[:, None]
    return linear.weight
