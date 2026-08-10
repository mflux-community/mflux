import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.ideogram4.ideogram4_initializer import Ideogram4Initializer
from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear


def _fp8_linear_with_lora(out_dims: int, in_dims: int) -> nn.Module:
    layer = Fp8Linear(in_dims, out_dims, bias=True)
    layer.weight = mx.random.randint(0, 255, (out_dims, in_dims)).astype(mx.uint8)
    layer.weight_scale = mx.random.uniform(0.5, 1.5, (out_dims,)).astype(mx.float32)
    layer.bias = mx.random.normal((out_dims,)).astype(mx.bfloat16)
    lora = LoRALinear.from_linear(layer, r=4, scale=1.0)
    lora.lora_A = mx.random.normal((in_dims, 4)).astype(mx.float32) * 0.05
    lora.lora_B = mx.random.normal((4, out_dims)).astype(mx.float32) * 0.05
    return lora


@pytest.mark.fast
def test_q8_folded_fp8_layer_survives_save_and_load():
    mx.random.seed(0)
    out_dims, in_dims = 8, 128

    class Holder(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.qkv = layer

    baked_holder = Holder(_fp8_linear_with_lora(out_dims, in_dims))
    LoRASaver.bake_and_strip_lora(baked_holder)
    assert isinstance(baked_holder.qkv, nn.QuantizedLinear), "fp8+LoRA bake should fold to q8"

    # What a native save stores for this layer, and what a fresh model must load.
    tree = {"qkv": dict(baked_holder.qkv.parameters())}
    fresh_holder = Holder(Fp8Linear(in_dims, out_dims, bias=True))

    # Without the rebuild the update silently misses and the forward raises.
    Ideogram4Initializer._rebuild_q8_folded_layers(fresh_holder, tree)
    assert isinstance(fresh_holder.qkv, nn.QuantizedLinear)
    fresh_holder.update(tree, strict=True)

    x = mx.random.normal((2, in_dims)).astype(mx.bfloat16)
    out_baked = baked_holder.qkv(x)
    out_loaded = fresh_holder.qkv(x)
    assert mx.allclose(out_baked, out_loaded, atol=1e-4, rtol=1e-3), "loaded q8 layer must match the baked one"


@pytest.mark.fast
def test_rebuild_leaves_plain_fp8_checkpoints_untouched():
    layer = Fp8Linear(64, 8, bias=True)
    holder_tree = {"qkv": {"weight": layer.weight, "weight_scale": layer.weight_scale, "bias": layer.bias}}

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = Fp8Linear(64, 8, bias=True)

    holder = Holder()
    Ideogram4Initializer._rebuild_q8_folded_layers(holder, holder_tree)
    assert isinstance(holder.qkv, Fp8Linear), "fp8 checkpoints must not be rewritten"
