import math

import mlx.core as mx
from mlx import nn


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def _timestep_embedding(self, t: mx.array, dim: int, max_period: int = 10) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half).astype(mx.float32) / half)
        args = t[..., None].astype(mx.float32) * freqs[None, ...]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self._timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.linear_1(t_freq)
        t_emb = nn.silu(t_emb)
        return self.linear_2(t_emb)
