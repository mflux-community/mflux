import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


@pytest.mark.fast
def test_new_lora_linear_matches_base_linear():
    linear = nn.Linear(8, 4, bias=False)
    x = mx.random.uniform(shape=(2, 8))

    wrapped = LoRALinear.from_linear(linear, r=4, scale=1.0)

    assert mx.array_equal(wrapped(x), linear(x)).item()


@pytest.mark.fast
def test_new_lora_linear_has_zero_b_and_random_a():
    wrapped = LoRALinear(input_dims=8, output_dims=4, r=4)

    assert mx.all(wrapped.lora_B == 0).item()
    assert mx.any(wrapped.lora_A != 0).item()
