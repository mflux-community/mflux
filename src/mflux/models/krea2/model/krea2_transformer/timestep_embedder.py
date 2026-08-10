import math

import mlx.core as mx
from mlx import nn


class Krea2TimestepMLP(nn.Module):
    def __init__(self, tdim: int, features: int):
        super().__init__()
        self.linear_in = nn.Linear(tdim, features)
        self.linear_out = nn.Linear(features, features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_out(nn.gelu_approx(self.linear_in(x)))

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> mx.array:
        t = time_factor * t
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb


class Krea2TimestepProj(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.linear = nn.Linear(features, features * 6)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(nn.gelu_approx(x))
