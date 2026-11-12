import mlx.core as mx
from mlx import nn


class SenseNovaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        normed = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        return self.weight * normed.astype(x.dtype)
