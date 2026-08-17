import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_transformer.bria_fibo_timesteps import BriaFiboTimesteps
from mflux.models.flux.model.flux_transformer.timestep_embedder import TimestepEmbedder


class BriaFiboTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int = 3072, time_theta: int = 10000):
        super().__init__()
        self.time_proj = BriaFiboTimesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, time_theta=time_theta, scale=1.0)  # fmt: off
        self.timestep_embedder = TimestepEmbedder(dim=embedding_dim)

    def __call__(self, timestep: mx.array, dtype) -> mx.array:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(dtype))
        return timesteps_emb
