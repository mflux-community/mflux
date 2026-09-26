import mlx.core as mx

from mflux.models.common.schedulers import register_contrib
from mflux.models.common.schedulers.base_scheduler import BaseScheduler
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler


class ViggleTurboScheduler(BaseScheduler):
    """Fixed sigma schedule shipped with Viggle's Qwen-Image-2.1-viggle-turbo LoRA.

    The DMD-distilled student was trained on the raw sigma nodes
    SIGMA_NODES = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25] (model card v0.2.1: "sample with
    sigmas=[...] in 6 transformer passes, no CFG" — the pipeline then applies its
    resolution-dependent time shift to them exactly as it does to its default nodes).
    This scheduler reproduces that: raw nodes + the shared LinearScheduler shift, with
    the terminal rescale OFF (the card requires shift_terminal=None: the base config's
    0.02 terminal "would wreck the last step"). step() is the same first-order Euler
    update LinearScheduler uses; only the node positions differ. The transformer's
    timestep routing (sigmas[t] as the model timestep) needs no change.

    num_inference_steps must be 6. Per the model card, alternative step counts (5/7) may
    only add or remove high-noise steps; that variant is left to the external-scheduler
    path. img2img/edit strength reuses the shared init_time_step machinery: strength < 1
    starts the loop at the node int(6 * strength) on this same shifted table.
    """

    SIGMA_NODES = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)

    def __init__(self, config):
        self.config = config
        steps = config.num_inference_steps
        if steps != len(self.SIGMA_NODES):
            raise ValueError(
                f"the viggle_turbo scheduler samples the distilled LoRA on its {len(self.SIGMA_NODES)} "
                f"trained sigma nodes, but num_inference_steps is {steps}. Use --steps {len(self.SIGMA_NODES)}."
            )
        self._sigmas = LinearScheduler.shift_sigmas(
            mx.array(list(self.SIGMA_NODES), dtype=mx.float32), config, use_terminal=False
        )
        self._timesteps = mx.arange(steps, dtype=mx.float32)

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
        return latents + noise.astype(latents.dtype) * dt


register_contrib(ViggleTurboScheduler, "viggle_turbo")
register_contrib(ViggleTurboScheduler, "ViggleTurboScheduler")
