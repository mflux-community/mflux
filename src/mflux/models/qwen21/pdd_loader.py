from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping


class Qwen21PDDLoader:
    FORMAT = "qwenimage21_extracted_prefused_v1"
    SIGMAS = mx.array([1.0, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471, 0.0])

    @staticmethod
    def is_pdd_adapter(lora_file: str) -> bool:
        _, metadata = mx.load(lora_file, return_metadata=True)
        return metadata.get("format") == Qwen21PDDLoader.FORMAT

    @staticmethod
    def load_and_apply(transformer: nn.Module, lora_file: str, scale: float, bake_lora: bool) -> None:
        if scale != 1.0:
            raise ValueError("Qwen Image 2.1 Fun Acc adapters require a LoRA scale of 1.0.")

        weights, metadata = mx.load(lora_file, return_metadata=True)
        if metadata.get("format") != Qwen21PDDLoader.FORMAT:
            raise ValueError(f"{Path(lora_file).name} is not a supported Qwen Image 2.1 PDD adapter.")

        print(f"Applying Qwen Image 2.1 PDD adapter: {Path(lora_file).name}")
        lora_weights = {key: value for key, value in weights.items() if ".lora_" in key}
        direct_weights = {key: value for key, value in weights.items() if ".lora_" not in key}
        mappings = LoRALoader._build_pattern_mappings(Qwen21LoRAMapping.get_mapping())
        applied_count, matched_keys, failed_targets = LoRALoader._apply_lora_with_mapping(
            transformer, lora_weights, scale, mappings, role=None
        )
        if failed_targets:
            raise ValueError(f"Could not apply PDD LoRA targets: {', '.join(failed_targets)}")
        unmatched_lora = set(lora_weights) - matched_keys
        if unmatched_lora:
            raise ValueError(f"Unsupported PDD LoRA tensors: {', '.join(sorted(unmatched_lora)[:5])}")

        for key, value in direct_weights.items():
            if key == "proj_out.weight":
                transformer.enable_pdd(value, Qwen21PDDLoader.SIGMAS)
                continue
            if not key.endswith(".weight"):
                raise ValueError(f"Unsupported PDD tensor: {key}")
            target_path = key[: -len(".weight")]
            target = LoRALoader._get_target_module(transformer, target_path)
            target_weight = getattr(target, "weight", None)
            if target_weight is None or target_weight.shape != value.shape:
                model_shape = None if target_weight is None else target_weight.shape
                raise ValueError(
                    f"PDD weight shape mismatch at {key}: model has {model_shape}, adapter has {value.shape}."
                )
            target.weight = value.astype(target_weight.dtype)

        if not transformer.has_pdd:
            raise ValueError("PDD adapter does not contain the required step-specific output heads.")

        print(f"Applied to {applied_count + len(direct_weights)} targets ({len(weights)}/{len(weights)} keys matched)")
        if bake_lora:
            print("Baking LoRA weights into the base model for faster inference...")
            LoRASaver.bake_and_strip_lora(transformer)
            mx.eval(transformer.parameters())
            print("LoRA weights baked successfully")
