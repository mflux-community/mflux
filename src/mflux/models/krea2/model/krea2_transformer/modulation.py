import mlx.core as mx
from mlx import nn


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = mx.zeros((6 * dim,))

    def __call__(self, vec: mx.array) -> tuple[mx.array, ...]:
        out = vec + self.lin.astype(vec.dtype)
        return mx.split(out, 6, axis=-1)


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = mx.zeros((2, dim))

    def __call__(self, vec: mx.array) -> tuple[mx.array, mx.array]:
        out = vec + self.lin.astype(vec.dtype)[None]
        scale, shift = mx.split(out, 2, axis=1)
        return scale, shift
