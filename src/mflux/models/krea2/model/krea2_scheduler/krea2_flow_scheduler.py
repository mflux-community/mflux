import mlx.core as mx

from mflux.models.common.schedulers.linear_scheduler import LinearScheduler


class Krea2FlowScheduler(LinearScheduler):
    """Sigma schedule for Krea 2.

    Krea 2 ships the Flux-style dynamic exponential time shift (scheduler_config.json:
    time_shift_type "exponential", use_dynamic_shifting true, base_shift 0.5, max_shift 1.15,
    image seq len 256..6400), which is exactly what the shared LinearScheduler computes from
    the ModelConfig sigma fields. Denoising itself is driven by the Krea2Sampler steppers
    (er_sde / euler), so step() is intentionally unsupported here.
    """

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        raise NotImplementedError("Krea-2 uses Krea2Sampler steppers during the denoise loop.")
