import math

import mlx.core as mx
from mlx import nn


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int = 256,
        hidden_dim: int = 2560,
        output_dim: int = 15360,
    ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def __call__(self, timestep: mx.array) -> mx.array:
        timestep = timestep[None] if timestep.ndim == 0 else timestep
        emb = TimeEmbedding._get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
        ).astype(self.proj_in.bias.dtype)
        emb = self.proj_in(emb)
        emb = nn.silu(emb)
        emb = self.proj_hid(emb)
        emb = nn.silu(emb)
        emb = self.proj_out(emb)
        return emb

    @staticmethod
    def _get_timestep_embedding(
        timesteps: mx.array,
        embedding_dim: int,
    ) -> mx.array:
        half_dim = embedding_dim // 2
        freqs = mx.exp(mx.arange(half_dim, dtype=mx.float32) * (-math.log(10000) / half_dim))
        args = timesteps[:, None].astype(mx.float32) * freqs
        return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
