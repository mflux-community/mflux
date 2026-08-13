from mflux.models.common.schedulers import register_contrib
from mflux.models.krea2.model.krea2_scheduler.krea2_flow_scheduler import Krea2FlowScheduler

register_contrib(Krea2FlowScheduler, "er_sde")
register_contrib(Krea2FlowScheduler, "euler")

__all__ = ["Krea2FlowScheduler"]
