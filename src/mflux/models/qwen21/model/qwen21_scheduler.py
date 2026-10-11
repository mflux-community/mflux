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

    @staticmethod
    def check_args(parser, args) -> None:
        # Runs before the model load, so a wrong --steps fails fast.
        if args.scheduler != "viggle_turbo":
            return
        nodes = len(ViggleTurboScheduler.SIGMA_NODES)
        if args.steps != nodes:
            # A replayed sidecar can set the scheduler without --scheduler on the command line.
            source = (
                "--scheduler viggle_turbo"
                if parser._option_was_provided("--scheduler")
                else "viggle_turbo (replayed from --config-from-conf)"
            )
            parser.error(
                f"{source} samples the distilled LoRA on its fixed sigma nodes; use --steps {nodes}, got {args.steps}"
            )
        if not args.lora_paths:
            print(
                "⚠️  --scheduler viggle_turbo without --lora runs the BASE model on 6 nodes; "
                "pass the distilled adapter for turbo results."
            )

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


class Qwen21TurboScheduler(BaseScheduler):
    """Fixed 8-step schedule of the Qwen-Image-2.1-Turbo checkpoint.

    The checkpoint saves its sigma nodes as `sample_sigmas` in model_index.json, and its
    scheduler config turns dynamic shifting off (shift 1.0, no shift_terminal). So this
    scheduler uses the nodes raw, with no resolution shift and no terminal rescale. The reference
    pipeline uses these nodes whatever num_inference_steps is, so any other step count
    is an error here. step() is the same first-order Euler update LinearScheduler uses.

    The Turbo model selects this schedule for the default "linear" scheduler name. See
    for_model().
    """

    SIGMA_NODES = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)

    def __init__(self, config):
        self.config = config
        steps = config.num_inference_steps
        if steps != len(self.SIGMA_NODES):
            raise ValueError(
                f"Qwen-Image-2.1-Turbo samples on its {len(self.SIGMA_NODES)} saved sigma nodes, "
                f"but num_inference_steps is {steps}. Use {len(self.SIGMA_NODES)} steps."
            )
        self._sigmas = mx.array([*self.SIGMA_NODES, 0.0], dtype=mx.float32)
        self._timesteps = mx.arange(steps, dtype=mx.float32)

    @staticmethod
    def is_turbo(model_config) -> bool:
        from mflux.models.common.config.model_config import ModelConfig

        # Every registry entry on the Turbo checkpoint, the ControlNet entry too.
        return model_config.model_name == ModelConfig.qwen_image_21_turbo().model_name

    @staticmethod
    def default_steps(model_config) -> int:
        # The step count when a Python API caller gives none: 8 for Turbo, else the base default.
        from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS

        if Qwen21TurboScheduler.is_turbo(model_config):
            return len(Qwen21TurboScheduler.SIGMA_NODES)
        return MODEL_INFERENCE_STEPS["qwen-image-2.1"]

    @staticmethod
    def for_model(model_config, scheduler: str) -> str:
        # The scheduler name Config should build. The Turbo checkpoint's default schedule is
        # its saved nodes, so "linear" (the default name) maps to them. The caller records
        # the original name in the image metadata, so a replay passes "linear" again.
        if scheduler == "linear" and Qwen21TurboScheduler.is_turbo(model_config):
            return "qwen21_turbo"
        return scheduler

    @staticmethod
    def check_args(parser, args, model_config) -> None:
        # Runs before the model load, so a wrong value fails fast.
        if not Qwen21TurboScheduler.is_turbo(model_config):
            return
        nodes = len(Qwen21TurboScheduler.SIGMA_NODES)
        if args.steps != nodes and not parser._option_was_provided("--steps"):
            # The default came from the checkpoint's name, and a local folder such as
            # Qwen--Qwen-Image-2.1-Turbo-mflux-q8 matches the "qwen" alias (20 steps) first.
            args.steps = nodes
        if args.steps != nodes:
            parser.error(f"Qwen-Image-2.1-Turbo samples on {nodes} fixed sigma nodes. Got --steps {args.steps}. Use --steps {nodes}.")  # fmt: off
        if args.guidance is not None and args.guidance != 1:
            parser.error(f"Qwen-Image-2.1-Turbo runs without CFG. Got --guidance {args.guidance}. Use --guidance 1.")
        if args.scheduler == "viggle_turbo":
            parser.error("viggle_turbo is the schedule of a LoRA for the base Qwen-Image-2.1. Do not use it with the Turbo model.")  # fmt: off

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
        return latents + noise.astype(latents.dtype) * dt


register_contrib(Qwen21TurboScheduler, "qwen21_turbo")
register_contrib(Qwen21TurboScheduler, "Qwen21TurboScheduler")
