import mlx.core as mx

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


class Qwen21PDDScheduler(BaseScheduler):
    def __init__(self, sigmas: mx.array):
        self._sigmas = sigmas.astype(mx.float32)

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return latents.astype(mx.float32) + noise.astype(mx.float32) * dt
