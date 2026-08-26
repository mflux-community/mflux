import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_saver import LoRASaver


def _quantized_base_with_small_adapter(bits: int):
    # Weight scale and delta scale mirror the real failure (#665): a q4 group step
    # measures ~10x a rank-32 LoRA delta, so requantizing at 4 bits rounds the
    # delta away while 8 bits carries it.
    mx.random.seed(0)
    linear = nn.Linear(256, 128, bias=False)
    linear.weight = mx.random.normal((128, 256)) * 0.04
    quantized = linear.to_quantized(group_size=64, bits=bits)
    lora = LoRALinear.from_linear(quantized, r=8)
    lora.lora_A = mx.random.normal((256, 8)) * 0.02
    lora.lora_B = mx.random.normal((8, 128)) * 0.02
    return quantized, lora


def test_bake_into_q4_upgrades_to_q8_so_the_delta_survives():
    quantized, lora = _quantized_base_with_small_adapter(bits=4)
    x = mx.random.normal((4, 256))
    expected = lora(x)  # runtime adapter: quantized base + dense delta

    baked = LoRASaver._bake_lora_into_linear(quantized, lora)

    assert isinstance(baked, nn.QuantizedLinear)
    assert baked.bits == 8
    base_err = float(mx.abs(quantized(x) - expected).mean())
    baked_err = float(mx.abs(baked(x) - expected).mean())
    # Requantizing the merge at 4 bits leaves baked_err at base_err (the adapter is
    # gone); at 8 bits the baked layer tracks the runtime-adapter output.
    assert baked_err < base_err * 0.5


def test_bake_into_q8_preserves_the_base_precision():
    quantized, lora = _quantized_base_with_small_adapter(bits=8)

    baked = LoRASaver._bake_lora_into_linear(quantized, lora)

    assert isinstance(baked, nn.QuantizedLinear)
    assert baked.bits == 8
    assert baked.group_size == 64


def test_bake_and_strip_reports_upgraded_layers(capsys):
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            quantized, lora = _quantized_base_with_small_adapter(bits=4)
            self.proj = lora

    transformer = Transformer()
    LoRASaver.bake_and_strip_lora(transformer)

    assert isinstance(transformer.proj, nn.QuantizedLinear)
    assert transformer.proj.bits == 8
    assert "Re-quantized 1 sub-8-bit layers at q8" in capsys.readouterr().out
