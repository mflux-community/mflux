import math

import mlx.core as mx
import numpy as np

from mflux.models.anima.model.anima_scheduler.anima_scheduler import AnimaScheduler


class AnimaSampler:
    # er_sde is the model card's default sampler. euler is the other one it names for Turbo and Aesthetic.
    NAMES = ("er_sde", "euler")
    DEFAULT = "er_sde"

    @staticmethod
    def resolve(name: str | None) -> str:
        # "linear" is the parser's generic --scheduler default and means "not chosen".
        if name is None or name == "linear":
            return AnimaSampler.DEFAULT
        if name not in AnimaSampler.NAMES:
            raise ValueError(f"Unknown Anima sampler {name!r}. Expected one of {', '.join(AnimaSampler.NAMES)}.")
        return name

    @staticmethod
    def make(name: str, sigmas: mx.array, seed: int) -> "AnimaEulerSampler | AnimaErSdeSampler":
        if AnimaSampler.resolve(name) == "euler":
            return AnimaEulerSampler(sigmas)
        return AnimaErSdeSampler(sigmas, seed)


class AnimaEulerSampler:
    def __init__(self, sigmas: mx.array):
        self.sigmas = sigmas

    def sigma(self, i: int) -> mx.array:
        return self.sigmas[i]

    def step(self, i: int, x: mx.array, velocity: mx.array, denoised: mx.array) -> mx.array:
        return x + (self.sigmas[i + 1] - self.sigmas[i]) * velocity


class AnimaErSdeSampler:
    # ComfyUI's er_sde (ER-SDE-Solver-3, arXiv 2309.06169) for a rectified-flow model, where
    # lambda = sigma / (1 - sigma) and alpha = 1 - sigma. As in the reference, the first sigma moves from 1 to
    # the point 1e-4 below it on the schedule, where lambda is finite, and the SDE noise has its own stream.
    NUM_POINTS = 200
    FIRST_SIGMA_OFFSET = 1e-4

    def __init__(self, sigmas: mx.array, seed: int, s_noise: float = 1.0, max_stage: int = 3):
        sig = np.array(sigmas, dtype=np.float32)
        if sig[0] >= 1.0:
            sig[0] = AnimaScheduler.shift(np.float32(1.0 - AnimaErSdeSampler.FIRST_SIGMA_OFFSET))
        self._sigmas = [float(s) for s in sig]
        self._lambdas = [s / (1.0 - s) for s in self._sigmas]
        self.s_noise = s_noise
        self.max_stage = max_stage
        self._key = mx.random.key(seed ^ 0xA41A)
        self._points = np.arange(AnimaErSdeSampler.NUM_POINTS, dtype=np.float64)
        self._old_denoised: mx.array | None = None
        self._old_denoised_d: mx.array | None = None

    def sigma(self, i: int) -> mx.array:
        return mx.array(self._sigmas[i], dtype=mx.float32)

    def step(self, i: int, x: mx.array, velocity: mx.array, denoised: mx.array) -> mx.array:
        if self._sigmas[i + 1] == 0:
            self._old_denoised = denoised
            return denoised
        lambdas, scale = self._lambdas, AnimaErSdeSampler._noise_scaler
        ls, lt = lambdas[i], lambdas[i + 1]
        alpha_s, alpha_t = self._sigmas[i] / ls, self._sigmas[i + 1] / lt
        r = scale(lt) / scale(ls)
        x = (alpha_t / alpha_s * r) * x + (alpha_t * (1 - r)) * denoised

        stage = min(self.max_stage, i + 1)
        if stage >= 2 and self._old_denoised is not None:
            dt = lt - ls
            step_size = -dt / AnimaErSdeSampler.NUM_POINTS
            positions = lt + self._points * step_size
            scaled = positions * (np.exp(positions**0.3) + 10.0)
            s = float(np.sum(1 / scaled)) * step_size
            denoised_d = (denoised - self._old_denoised) / (ls - lambdas[i - 1])
            x = x + (alpha_t * (dt + s * scale(lt))) * denoised_d
            if stage >= 3 and self._old_denoised_d is not None:
                s_u = float(np.sum((positions - ls) / scaled)) * step_size
                denoised_u = (denoised_d - self._old_denoised_d) / ((ls - lambdas[i - 2]) / 2)
                x = x + (alpha_t * ((dt**2) / 2 + s_u * scale(lt))) * denoised_u
            self._old_denoised_d = denoised_d

        if self.s_noise > 0:
            std = math.sqrt(max(lt**2 - ls**2 * r**2, 0.0))
            x = x + self._noise(x.shape) * (alpha_t * self.s_noise * std)
        self._old_denoised = denoised
        return x

    def _noise(self, shape: tuple[int, ...]) -> mx.array:
        self._key, key = mx.random.split(self._key)
        return mx.random.normal(shape, key=key)

    @staticmethod
    def _noise_scaler(x: float) -> float:
        return x * (math.exp(x**0.3) + 10.0)
