import math

import mlx.core as mx
from mlx import nn


def get_timestep_embedding(timesteps: mx.array, dim: int) -> mx.array:
    half_dim = dim // 2
    freq = math.log(10000) / half_dim
    freqs = mx.exp(-freq * mx.arange(half_dim, dtype=mx.float32))
    args = timesteps[:, None].astype(mx.float32) * freqs[None, :]
    return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)


class ErnieTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, t_emb: mx.array) -> mx.array:
        x = self.linear_1(t_emb)
        x = nn.silu(x)
        return self.linear_2(x)
