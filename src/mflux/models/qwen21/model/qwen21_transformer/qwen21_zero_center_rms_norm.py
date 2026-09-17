import mlx.core as mx
from mlx import nn


class Qwen21ZeroCenterRMSNorm(nn.Module):
    # RMSNorm whose checkpoint weight is stored zero-centered: the effective scale is weight + 1.

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x_float = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(x_float * x_float, axis=-1, keepdims=True) + self.eps)
        return (x_float * rrms * (self.weight.astype(mx.float32) + 1)).astype(input_dtype)
