import math

import mlx.core as mx
from mlx import nn


class SenseNovaTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int = 4096, freq_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = [nn.Linear(freq_dim, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True)]  # fmt: skip

    def __call__(self, t: mx.array) -> mx.array:
        # t: [B] used as is (no 1000x); returns [B, hidden_size]
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(self.max_period) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(self.mlp[0].weight.dtype)
        return self.mlp[2](self.mlp[1](self.mlp[0](emb)))
