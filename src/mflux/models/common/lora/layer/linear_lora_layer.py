import math

import mlx.core as mx
from mlx import nn


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        r: int = 16,
        scale: float = 1.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale
        scale = 1 / math.sqrt(input_dims)

        self.lora_A = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        # Zero B so that a new adapter starts as the identity (Hu et al. 2021, section 4.1).
        self.lora_B = mx.zeros((r, output_dims))

    def __call__(self, x):
        base_out = self.linear(x)
        # Cast the float32 master factors to the activation dtype. Without this cast,
        # MLX promotes the output to float32 and every later block runs in float32.
        lora_out = mx.matmul(mx.matmul(x, self.lora_A.astype(x.dtype)), self.lora_B.astype(x.dtype))
        return base_out + self.scale * lora_out
