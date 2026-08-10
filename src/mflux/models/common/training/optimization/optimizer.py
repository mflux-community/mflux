from __future__ import annotations

from enum import Enum
from pathlib import Path

import mlx.core as mx
import mlx.optimizers
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.zip_util import ZipUtil


class Optimizers(Enum):
    ADAM = ("Adam", optim.Adam)
    ADAMW = ("AdamW", optim.AdamW)
    ADAFACTOR = ("Adafactor", optim.Adafactor)  # ~half the optimizer state of Adam (memory-light)
    LION = ("Lion", optim.Lion)

    def __init__(self, alias: str, optimizer: mlx.optimizers.Optimizer):
        self.alias = alias
        self.optimizer = optimizer

    @staticmethod
    def from_alias(alias: str) -> "mlx.optimizers.Optimizer":
        for opt in Optimizers:
            if opt.alias == alias:
                return opt.optimizer
        raise ValueError(f"'{alias}' is not a valid optimizer")


class Optimizer:
    def __init__(self, optimizer: mlx.optimizers.Optimizer):
        self.optimizer = optimizer

    def save(self, path: Path) -> None:
        state = tree_flatten(self.optimizer.state)
        mx.save_safetensors(str(path), dict(state))

    @staticmethod
    def _build_lr(spec):
        # Returns a float (constant) or an MLX schedule callable. A linear warmup (lr_warmup_steps)
        # is prepended when set; "cosine" decays over the remaining steps (needs lr_total_steps).
        lr = spec.learning_rate
        warmup = spec.lr_warmup_steps or 0
        if spec.lr_schedule == "cosine" and spec.lr_total_steps:
            decay_steps = max(1, int(spec.lr_total_steps) - warmup)
            main = optim.cosine_decay(lr, decay_steps)
            if warmup > 0:
                return optim.join_schedules([optim.linear_schedule(0.0, lr, warmup), main], [warmup])
            return main
        if warmup > 0:  # warmup then constant
            return optim.join_schedules([optim.linear_schedule(0.0, lr, warmup), optim.cosine_decay(lr, 10**12)], [warmup])
        return lr

    @staticmethod
    def from_spec(training_spec: TrainingSpec) -> "Optimizer":
        opt_cls = Optimizers.from_alias(training_spec.optimizer.name)
        # noinspection PyCallingNonCallable
        opt = opt_cls(learning_rate=Optimizer._build_lr(training_spec.optimizer))

        if training_spec.optimizer.state_path is not None:
            state = ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.optimizer.state_path,
                loader=lambda x: tree_unflatten(list(mx.load(x).items())),
            )
            opt.state = state

        return Optimizer(opt)
