import mlx.core as mx
import numpy as np


class MOSSImageScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, training_shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.training_shift = training_shift
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int = 50):
        total = self.num_train_timesteps

        def grid(count, s):
            lower = s / (total + s - 1)
            values = np.linspace(1.0, lower, count)
            return s * values / (1.0 + (s - 1.0) * values)

        sigmas = grid(num_inference_steps, self.shift)
        training = grid(total, self.training_shift)[::-1]

        # searchsorted
        upper = np.clip(np.searchsorted(training, sigmas), 0, total - 1)
        lower = np.clip(upper - 1, 0, total - 1)

        diff_lower = np.abs(sigmas - training[lower])
        diff_upper = np.abs(training[upper] - sigmas)
        nearest = np.where(diff_lower <= diff_upper, lower, upper)

        # unique consecutive
        snapped = training[nearest]
        mask = np.concatenate([[True], snapped[1:] != snapped[:-1]])
        sigmas = snapped[mask]

        # Append terminal 0.0
        sigmas = np.concatenate([sigmas, [0.0]])
        self.sigmas = mx.array(sigmas, dtype=mx.float32)
        self.timesteps = self.sigmas[:-1] * 1000.0
        return self.timesteps

    def step(self, model_output: mx.array, timestep_idx: int, sample: mx.array) -> mx.array:
        dt = self.sigmas[timestep_idx + 1] - self.sigmas[timestep_idx]
        prev_sample = sample + model_output * dt
        return prev_sample
