from __future__ import annotations

from typing import Any

import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.lora.path_util import (
    expand_module_paths_from_targets,
    get_at_path,
    set_at_path,
)
from mflux.models.common.training.state.training_spec import LoraTargetSpec


def inject_lora_targets(transformer: Any, targets: list[LoraTargetSpec]) -> None:
    expanded = expand_module_paths_from_targets(targets)
    for module_path, rank, alpha in expanded:
        scale = (alpha / rank) if alpha is not None else 1.0
        current = get_at_path(transformer, module_path)

        # Skip if already has a trainable LoRA on this path
        if isinstance(current, LoRALinear):
            if getattr(current, "_mflux_lora_role", None) == "train":
                continue
        if isinstance(current, FusedLoRALinear):
            if any(getattr(lora, "_mflux_lora_role", None) == "train" for lora in current.loras):
                continue

        if isinstance(current, (nn.Linear, nn.QuantizedLinear)):
            wrapped = LoRALinear.from_linear(current, r=rank, scale=scale)
            wrapped._mflux_lora_role = "train"
            set_at_path(transformer, module_path, wrapped)
        elif isinstance(current, LoRALinear):
            # Fuse a new trainable LoRA on top of an existing LoRA (e.g. assistant adapter).
            train_lora = LoRALinear.from_linear(current.linear, r=rank, scale=scale)
            train_lora._mflux_lora_role = "train"
            fused = FusedLoRALinear(base_linear=current.linear, loras=[current, train_lora])
            set_at_path(transformer, module_path, fused)
        elif isinstance(current, FusedLoRALinear):
            # Add a new trainable LoRA to an existing fusion (e.g. multiple preloaded LoRAs).
            train_lora = LoRALinear.from_linear(current.base_linear, r=rank, scale=scale)
            train_lora._mflux_lora_role = "train"
            fused = FusedLoRALinear(base_linear=current.base_linear, loras=current.loras + [train_lora])
            set_at_path(transformer, module_path, fused)
        else:
            raise TypeError(
                f"LoRA target '{module_path}' must resolve to nn.Linear or nn.QuantizedLinear, got {type(current)}"
            )
