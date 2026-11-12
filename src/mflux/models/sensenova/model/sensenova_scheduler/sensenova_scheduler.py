import mlx.core as mx
import numpy as np


class SenseNovaScheduler:
    @staticmethod
    def check_steps(num_inference_steps: int) -> None:
        if num_inference_steps < 1:
            raise ValueError(f"the step count must be at least 1, got {num_inference_steps}")

    def __init__(self, num_inference_steps: int, shift: float = 3.0):
        t = np.linspace(0.0, 1.0, num_inference_steps + 1, dtype=np.float64)
        sigma = shift * (1.0 - t) / (1.0 + (shift - 1.0) * (1.0 - t))
        self.timesteps = [float(1.0 - s) for s in sigma]

    def model_time(self, step: int) -> mx.array:
        return mx.array([self.timesteps[step]], dtype=mx.float32)

    def step(self, latents: mx.array, velocity: mx.array, step: int) -> mx.array:
        return latents + (self.timesteps[step + 1] - self.timesteps[step]) * velocity
