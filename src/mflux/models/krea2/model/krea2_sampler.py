import mlx.core as mx


class Krea2Sampler:
    @staticmethod
    def flow_sigmas(num_steps: int, shift: float = 1.15) -> mx.array:
        sigmas = mx.linspace(1.0, 0.0, num_steps + 1)
        return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

    @staticmethod
    def make_stepper(name: str, sigmas: mx.array, seed: int):
        if name == "euler":
            return EulerStepper(sigmas)
        if name == "er_sde":
            return ErSdeStepper(sigmas, seed)
        raise ValueError(f"Unknown Krea-2 sampler '{name}' (expected 'er_sde' or 'euler').")


class EulerStepper:
    def __init__(self, sigmas: mx.array):
        self.sigmas = sigmas

    def step(self, i: int, x: mx.array, v: mx.array, denoised: mx.array) -> mx.array:
        return x + (self.sigmas[i + 1] - self.sigmas[i]) * v


class ErSdeStepper:
    NUM_POINTS = 200.0

    def __init__(self, sigmas: mx.array, seed: int, s_noise: float = 1.0, max_stage: int = 3):
        self.sigmas = sigmas
        self.s_noise = s_noise
        self.max_stage = max_stage
        self._key = mx.random.key(seed ^ 0x5DE)  # independent stream for SDE noise
        self._point_indice = mx.arange(0, self.NUM_POINTS, dtype=mx.float32)
        self._old_denoised: mx.array | None = None
        self._old_denoised_d: mx.array | None = None

    @staticmethod
    def _noise_scaler(x: mx.array) -> mx.array:
        return x * (mx.exp(x**0.3) + 10.0)

    def step(self, i: int, x: mx.array, v: mx.array, denoised: mx.array) -> mx.array:
        sigmas = self.sigmas
        if sigmas[i + 1].item() == 0:
            self._old_denoised = denoised
            return denoised

        ls, lt = sigmas[i], sigmas[i + 1]  # er_lambda = sigma (alpha = 1)
        r = self._noise_scaler(lt) / self._noise_scaler(ls)

        # Stage 1 (Euler in er_lambda space; r_alpha = 1)
        x = r * x + (1 - r) * denoised

        stage = min(self.max_stage, i + 1)
        if stage >= 2 and self._old_denoised is not None:
            dt = lt - ls
            step_size = -dt / self.NUM_POINTS
            lam_pos = lt + self._point_indice * step_size
            scaled = self._noise_scaler(lam_pos)

            s = mx.sum(1.0 / scaled) * step_size
            denoised_d = (denoised - self._old_denoised) / (ls - sigmas[i - 1])
            x = x + (dt + s * self._noise_scaler(lt)) * denoised_d

            if stage >= 3 and self._old_denoised_d is not None:
                s_u = mx.sum((lam_pos - ls) / scaled) * step_size
                denoised_u = (denoised_d - self._old_denoised_d) / ((ls - sigmas[i - 2]) / 2)
                x = x + ((dt**2) / 2 + s_u * self._noise_scaler(lt)) * denoised_u
            self._old_denoised_d = denoised_d

        # SDE noise injection (alpha_t = 1)
        if self.s_noise > 0:
            self._key, sub = mx.random.split(self._key)
            noise = mx.random.normal(x.shape, key=sub)
            var = lt**2 - (ls**2) * (r**2)
            std = mx.sqrt(mx.maximum(var, 0.0))
            x = x + noise * self.s_noise * std

        self._old_denoised = denoised
        return x
