import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config import ModelConfig


class Qwen21Timesteps(nn.Module):
    # Sinusoidal embedding with cos in the first half and sin in the second (the 2.1 order).

    def __init__(self, timestep_dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        self.freqs = mx.array(np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half))

    def __call__(self, timestep: mx.array) -> mx.array:
        args = timestep.astype(mx.float32)[:, None] * self.time_factor * self.freqs[None, :]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class Qwen21TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 4096):
        super().__init__()
        self.linear_1 = nn.Linear(256, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def __call__(self, timesteps_proj: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(timesteps_proj)))


class Qwen21TimeTextEmbed(nn.Module):
    def __init__(self, embedding_dim: int = 4096):
        super().__init__()
        self.time_proj = Qwen21Timesteps()
        self.timestep_embedder = Qwen21TimestepEmbedder(embedding_dim)

    def __call__(self, timestep: mx.array) -> mx.array:
        # compute the sinusoid in fp32, then follow the reference's cast to the model dtype
        # before the embedder — otherwise fp32 modulation promotes the whole hidden stream.
        # Cast by ModelConfig.precision, never by weight dtype: under quantization the
        # weights are packed uint32 and casting against them shreds the embedding.
        timesteps_proj = self.time_proj(timestep).astype(ModelConfig.precision)
        return self.timestep_embedder(timesteps_proj)
