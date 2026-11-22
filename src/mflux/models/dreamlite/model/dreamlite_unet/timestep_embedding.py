import math

import mlx.core as mx
from mlx import nn


class DreamLiteTimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, projected: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(projected)))

    @staticmethod
    def sinusoidal(values: mx.array, dim: int) -> mx.array:
        # diffusers' Timesteps with flip_sin_to_cos and no frequency shift: cosines first, then sines
        half = dim // 2
        frequencies = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        angles = values.astype(mx.float32)[:, None] * frequencies[None]
        return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)
