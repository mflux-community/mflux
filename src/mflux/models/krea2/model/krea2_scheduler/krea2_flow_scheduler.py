from typing import TYPE_CHECKING

import mlx.core as mx

from mflux.models.common.schedulers.base_scheduler import BaseScheduler
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config


class Krea2FlowScheduler(BaseScheduler):
    def __init__(self, config: "Config"):
        self.config = config
        self._sigmas = Krea2Sampler.flow_sigmas(
            config.num_inference_steps,
            config.model_config.sigma_max_shift,
        )

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        raise NotImplementedError("Krea-2 uses Krea2Sampler steppers during the denoise loop.")
