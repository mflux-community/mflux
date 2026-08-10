import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lokr_factors import is_lokr_adapter, rebuild_lokr_factors
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.common.resolution.lora_resolution import LoraResolution


@dataclass
class PatternMatch:
    source_pattern: str
    target_path: str
    matrix_name: str
    transpose: bool
    transform: Callable[[mx.array], mx.array] | None = None


class LoRALoader:
    @staticmethod
    def load_and_apply_lora(
        lora_mapping: list[LoRATarget],
        transformer: nn.Module,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        role: str | None = None,
        bake_lora: bool = True,
    ) -> tuple[list[str], list[float]]:
        resolved_paths = LoraResolution.resolve_paths(lora_paths)
        if not resolved_paths:
            return resolved_paths, []

        resolved_scales = LoraResolution.resolve_scales(lora_scales, len(resolved_paths))
        if len(resolved_scales) != len(resolved_paths):
            raise ValueError(
                f"Number of LoRA scales ({len(resolved_scales)}) must match number of LoRA files ({len(resolved_paths)})"
            )

        print(f"📦 Loading {len(resolved_paths)} LoRA file(s)...")

        for lora_file, scale in zip(resolved_paths, resolved_scales):
            LoRALoader._apply_single_lora(transformer, lora_file, scale, lora_mapping, role=role)

        print("✅ All LoRA weights applied successfully")

        if bake_lora:
            print("Baking LoRA weights into the base model for faster inference...")
            LoRASaver.bake_and_strip_lora(transformer)
            mx.eval(transformer.parameters())
            print("✅ LoRA weights baked successfully")

        return resolved_paths, resolved_scales

    @staticmethod
    def _apply_single_lora(
        transformer: nn.Module,
        lora_file: str,
        scale: float,
        lora_mapping: list[LoRATarget],
        *,
        role: str | None,
    ) -> None:
        # Load the LoRA weights
        if not Path(lora_file).exists():
            print(f"❌ LoRA file not found: {lora_file}")
            return

        print(f"🔧 Applying LoRA: {Path(lora_file).name} (scale={scale})")

        try:
            weights = dict(mx.load(lora_file, return_metadata=True)[0].items())
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"❌ Failed to load LoRA file: {e}")
            return

        # Build pattern mappings from LoRATargets
        pattern_mappings = LoRALoader._build_pattern_mappings(lora_mapping)

        # Apply LoRA using the mappings (allows multiple targets per source)
        applied_count, matched_keys = LoRALoader._apply_lora_with_mapping(
            transformer, weights, scale, pattern_mappings, role=role
        )

        # Report results
        total_keys = len(weights)
        unmatched_keys = set(weights.keys()) - matched_keys

        print(f"   ✅ Applied to {applied_count} layers ({len(matched_keys)}/{total_keys} keys matched)")

        if unmatched_keys:
            print(f"   ⚠️  {len(unmatched_keys)} unmatched keys in LoRA file:")
            for key in sorted(unmatched_keys)[:5]:
                print(f"      - {key}")
            if len(unmatched_keys) > 5:
                print(f"      ... and {len(unmatched_keys) - 5} more")

        if applied_count == 0:
            raise ValueError(f"No LoRA layers were applied from {Path(lora_file).name}")

    @staticmethod
    def _build_pattern_mappings(targets: list[LoRATarget]) -> list[PatternMatch]:
        mappings = []

        for target in targets:
            # Add up weight patterns (lora_B)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="lora_B",
                    transpose=True,
                    transform=target.up_transform,
                )
                for pattern in target.possible_up_patterns
            )

            # Add down weight patterns (lora_A)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="lora_A",
                    transpose=True,
                    transform=target.down_transform,
                )
                for pattern in target.possible_down_patterns
            )

            # Add alpha patterns (no transpose, no transform)
            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="alpha",
                    transpose=False,
                    transform=None,
                )
                for pattern in target.possible_alpha_patterns
            )

            mappings.extend(
                LoRALoader._lokr_factor_pattern_matches(
                    target.possible_lokr_w1_patterns,
                    target.model_path,
                    full_name="lokr_w1",
                    factor_names=("lokr_w1_a", "lokr_w1_b"),
                    transform=target.lokr_w1_transform,
                )
            )

            mappings.extend(
                LoRALoader._lokr_factor_pattern_matches(
                    target.possible_lokr_w2_patterns,
                    target.model_path,
                    full_name="lokr_w2",
                    factor_names=("lokr_w2_a", "lokr_w2_b"),
                    transform=target.lokr_w2_transform,
                    extra_factor_suffix="lokr_t2",
                )
            )

            mappings.extend(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target.model_path,
                    matrix_name="dora_scale",
                    transpose=False,
                    transform=None,
                )
                for pattern in target.possible_dora_scale_patterns
            )

        return mappings

    @staticmethod
    def _lokr_factor_pattern_matches(
        patterns: list[str],
        target_path: str,
        *,
        full_name: str,
        factor_names: tuple[str, str],
        transform: Callable[[mx.array], mx.array] | None,
        extra_factor_suffix: str | None = None,
    ) -> list[PatternMatch]:
        matches: list[PatternMatch] = []
        full_suffix = f".{full_name}"

        for pattern in patterns:
            matches.append(
                PatternMatch(
                    source_pattern=pattern,
                    target_path=target_path,
                    matrix_name=full_name,
                    transpose=False,
                    transform=transform,
                )
            )

            if not pattern.endswith(full_suffix):
                continue

            base = pattern[: -len(full_suffix)]
            matches.extend(
                PatternMatch(
                    source_pattern=f"{base}.{factor_name}",
                    target_path=target_path,
                    matrix_name=factor_name,
                    transpose=False,
                    transform=transform,
                )
                for factor_name in factor_names
            )

            if extra_factor_suffix is not None:
                matches.append(
                    PatternMatch(
                        source_pattern=f"{base}.{extra_factor_suffix}",
                        target_path=target_path,
                        matrix_name=extra_factor_suffix,
                        transpose=False,
                        transform=transform,
                    )
                )

        return matches

    @staticmethod
    def _apply_lora_with_mapping(
        transformer: nn.Module,
        weights: dict,
        scale: float,
        pattern_mappings: list[PatternMatch],
        *,
        role: str | None,
    ) -> tuple[int, set]:
        applied_count = 0
        lora_data_by_target: dict[str, dict] = {}
        matched_keys: set[str] = set()

        # For each weight key, find ALL matching patterns (not just first)
        # This allows multiple targets to use the same source (e.g., QKV split)
        for weight_key, weight_value in weights.items():
            for mapping in pattern_mappings:
                match_result = LoRALoader._match_pattern(weight_key, mapping.source_pattern)
                if match_result is None:
                    continue

                matched_keys.add(weight_key)
                block_idx = match_result

                # Resolve target path with block index if needed
                target_path = mapping.target_path
                if block_idx is not None and "{block}" in target_path:
                    target_path = target_path.format(block=block_idx)

                # Apply transform if specified
                transformed_value = weight_value
                if mapping.transform is not None:
                    transformed_value = mapping.transform(weight_value)

                # Apply transpose if needed
                if mapping.transpose:
                    transformed_value = transformed_value.T

                # Store for this target
                if target_path not in lora_data_by_target:
                    lora_data_by_target[target_path] = {}

                lora_data_by_target[target_path][mapping.matrix_name] = transformed_value

        # Apply LoRA to each target
        for target_path, lora_data in lora_data_by_target.items():
            if LoRALoader._apply_adapter_to_target(transformer, target_path, lora_data, scale, role=role):
                applied_count += 1

        return applied_count, matched_keys

    @staticmethod
    def _apply_adapter_to_target(
        transformer: nn.Module, target_path: str, lora_data: dict, scale: float, *, role: str | None
    ) -> bool:
        if is_lokr_adapter(lora_data):
            return LoRALoader._apply_lokr_matrices_to_target(transformer, target_path, lora_data, scale, role=role)

        return LoRALoader._apply_lora_matrices_to_target(transformer, target_path, lora_data, scale, role=role)

    @staticmethod
    def _match_pattern(weight_key: str, pattern: str) -> int | None:
        if "{block}" in pattern:
            # Find all numbers in the weight key
            numbers_in_key = re.findall(r"\d+", weight_key)
            for num_str in numbers_in_key:
                test_block_idx = int(num_str)
                concrete_pattern = pattern.replace("{block}", str(test_block_idx))
                if weight_key == concrete_pattern:
                    return test_block_idx
            return None
        else:
            if weight_key == pattern:
                return 0  # Return 0 to indicate match (no block)
            return None

    @staticmethod
    def _apply_lora_matrices_to_target(
        transformer: nn.Module, target_path: str, lora_data: dict, scale: float, *, role: str | None
    ) -> bool:
        try:
            current_module = LoRALoader._get_target_module(transformer, target_path)
        except (AttributeError, IndexError, KeyError):
            print(f"❌ Could not find target path: {target_path}")
            return False

        # Check if we have the required matrices
        if "lora_A" not in lora_data or "lora_B" not in lora_data:
            print(f"❌ Missing required LoRA matrices for {target_path}")
            return False

        # Values are already transformed and transposed
        lora_A = lora_data["lora_A"]
        lora_B = lora_data["lora_B"]

        # Handle alpha scaling
        alpha_scale = 1.0
        if "alpha" in lora_data:
            alpha_value = lora_data["alpha"]
            rank = lora_A.shape[1]
            alpha_scale = float(alpha_value) / rank

        # Calculate final scale - only use user scale, matching Diffusers approach
        effective_scale = scale

        # Create new LoRA layer
        # Check if it's a linear layer (either nn.Linear, LoRALinear, or FusedLoRALinear)
        is_linear = hasattr(current_module, "weight")
        is_lora_linear = isinstance(current_module, LoRALinear)
        is_lokr_linear = isinstance(current_module, LoKrLinear)
        is_fused_linear = isinstance(current_module, FusedLoRALinear)

        if is_linear or is_lora_linear or is_lokr_linear or is_fused_linear:
            # Handle fusion: if the current module is already a LoRA layer, fuse them
            if is_lora_linear or is_lokr_linear:
                print(f"   🔀 Fusing with existing LoRA at {target_path}")
                lora_layer = LoRALinear.from_linear(current_module.linear, r=lora_A.shape[1], scale=effective_scale)
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                fused_layer = FusedLoRALinear(base_linear=current_module.linear, loras=[current_module, lora_layer])
                replacement_layer = fused_layer
            elif is_fused_linear:
                print(f"   🔀 Adding to existing fusion at {target_path}")
                lora_layer = LoRALinear.from_linear(
                    current_module.base_linear, r=lora_A.shape[1], scale=effective_scale
                )
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                fused_layer = FusedLoRALinear(
                    base_linear=current_module.base_linear, loras=current_module.loras + [lora_layer]
                )
                replacement_layer = fused_layer
            else:
                # First LoRA on this layer
                lora_layer = LoRALinear.from_linear(current_module, r=lora_A.shape[1], scale=effective_scale)
                lora_layer._mflux_lora_role = role
                lora_layer.lora_A = lora_A
                lora_layer.lora_B = lora_B
                if "alpha" in lora_data:
                    lora_layer.lora_B = lora_layer.lora_B * alpha_scale
                replacement_layer = lora_layer

            LoRALoader._replace_target_module(transformer, target_path, replacement_layer)

            return True
        else:
            print(f"❌ Target layer {target_path} is not a linear layer")
            return False

    @staticmethod
    def _apply_lokr_matrices_to_target(
        transformer: nn.Module, target_path: str, lora_data: dict, scale: float, *, role: str | None
    ) -> bool:
        try:
            current_module = LoRALoader._get_target_module(transformer, target_path)
        except (AttributeError, IndexError, KeyError):
            print(f"❌ Could not find target path: {target_path}")
            return False

        try:
            lokr_w1, lokr_w2, alpha_scale = rebuild_lokr_factors(lora_data)
        except ValueError as error:
            raise ValueError(f"Invalid LoKr matrices for {target_path}: {error}") from error

        effective_scale = scale * alpha_scale

        if not hasattr(current_module, "weight") and not isinstance(
            current_module, (LoRALinear, FusedLoRALinear, LoKrLinear)
        ):
            print(f"❌ Target layer {target_path} is not a linear layer")
            return False

        if isinstance(current_module, FusedLoRALinear):
            base_linear = current_module.base_linear
            existing_loras = current_module.loras
        elif isinstance(current_module, (LoRALinear, LoKrLinear)):
            base_linear = current_module.linear
            existing_loras = [current_module]
        else:
            base_linear = current_module
            existing_loras = []

        try:
            replacement_layer = LoKrLinear.from_linear(
                base_linear,
                lokr_w1=lokr_w1,
                lokr_w2=lokr_w2,
                dora_scale=lora_data.get("dora_scale"),
                scale=effective_scale,
            )
        except ValueError as error:
            raise ValueError(f"Invalid LoKr matrices for {target_path}: {error}") from error

        replacement_layer._mflux_lora_role = role
        if existing_loras:
            print(f"   🔀 Adding LoKr to existing fusion at {target_path}")
            replacement_layer = FusedLoRALinear(base_linear=base_linear, loras=existing_loras + [replacement_layer])

        LoRALoader._replace_target_module(transformer, target_path, replacement_layer)
        return True

    @staticmethod
    def _get_target_module(transformer: nn.Module, target_path: str):
        current_module = transformer
        for part in target_path.split("."):
            if part.isdigit():
                current_module = current_module[int(part)]
            elif isinstance(current_module, dict) and part in current_module:
                current_module = current_module[part]
            else:
                current_module = getattr(current_module, part)

        return current_module

    @staticmethod
    def _replace_target_module(transformer: nn.Module, target_path: str, replacement_layer: nn.Module) -> None:
        path_parts = target_path.split(".")
        parent_module = transformer
        for part in path_parts[:-1]:
            if part.isdigit():
                parent_module = parent_module[int(part)]
            elif isinstance(parent_module, dict) and part in parent_module:
                parent_module = parent_module[part]
            else:
                parent_module = getattr(parent_module, part)

        final_attr = path_parts[-1]
        if final_attr.isdigit():
            parent_module[int(final_attr)] = replacement_layer
        elif isinstance(parent_module, dict) and final_attr in parent_module:
            parent_module[final_attr] = replacement_layer
        else:
            setattr(parent_module, final_attr, replacement_layer)
