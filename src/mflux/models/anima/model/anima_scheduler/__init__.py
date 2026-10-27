from mflux.models.anima.model.anima_scheduler.anima_sampler import AnimaSampler
from mflux.models.anima.model.anima_scheduler.anima_scheduler import AnimaScheduler
from mflux.models.common.schedulers import register_contrib

register_contrib(AnimaScheduler, "anima")

__all__ = ["AnimaSampler", "AnimaScheduler"]
