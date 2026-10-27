from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mflux.models.common.schedulers.base_scheduler import BaseScheduler

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config


class AnimaScheduler(BaseScheduler):
    # ComfyUI's "simple" schedule on Anima's flow sampling (shift 3, multiplier 1): every (1000 / steps)-th
    # point of the shifted 1000-step training grid, counted from the top, then 0. The model's timestep is
    # the sigma itself. step() is the Euler update; er_sde runs through AnimaErSdeSampler instead.
    NUM_TRAIN_TIMESTEPS = 1000
    SHIFT = 3.0

    def __init__(self, config: "Config"):
        self.config = config
        self._sigmas = AnimaScheduler.compute_sigmas(config.num_inference_steps)

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._sigmas[:-1]

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return latents.astype(mx.float32) + noise.astype(mx.float32) * dt

    @staticmethod
    def compute_sigmas(num_inference_steps: int) -> mx.array:
        AnimaScheduler.check_steps(num_inference_steps)
        total = AnimaScheduler.NUM_TRAIN_TIMESTEPS
        grid = AnimaScheduler.shift(np.arange(1, total + 1, dtype=np.float32) / np.float32(total))
        stride = total / num_inference_steps
        picks = [grid[-(1 + int(i * stride))] for i in range(num_inference_steps)]
        return mx.array(np.array([*picks, 0.0], dtype=np.float32))

    @staticmethod
    def check_steps(num_inference_steps: int) -> None:
        # The schedule picks from 1000 grid points: more steps would repeat a sigma, and er_sde divides by the gap.
        if not 1 <= num_inference_steps <= AnimaScheduler.NUM_TRAIN_TIMESTEPS:
            raise ValueError(f"Anima takes 1 to {AnimaScheduler.NUM_TRAIN_TIMESTEPS} steps, got {num_inference_steps}.")

    @staticmethod
    def shift(t: np.ndarray) -> np.ndarray:
        shift = np.float32(AnimaScheduler.SHIFT)
        return shift * t / (np.float32(1.0) + (shift - np.float32(1.0)) * t)
