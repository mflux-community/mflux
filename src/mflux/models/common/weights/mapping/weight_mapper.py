import re
from typing import Callable, Dict, List, Optional

import mlx.core as mx

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget


class WeightMapper:
    _PLACEHOLDERS = ("{block}", "{layer}", "{i}", "{res}")

    @staticmethod
    def apply_mapping(
        hf_weights: Dict[str, mx.array],
        mapping: List[WeightTarget],
        num_blocks: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> Dict:
        # Auto-detect number of blocks if not provided
        if num_blocks is None:
            num_blocks = WeightMapper._detect_num_blocks(hf_weights)

        # Auto-detect number of layers if not provided
        if num_layers is None:
            num_layers = WeightMapper._detect_num_layers(hf_weights)

        # Build flat mapping: HF pattern -> [(MLX path, transform), ...] (supports one-to-many)
        flat_mapping = WeightMapper._build_flat_mapping(mapping, num_blocks, num_layers)

        # Map weights
        mapped_weights = {}
        mapped_count = 0
        skipped_count = 0

        for hf_key, hf_tensor in hf_weights.items():
            # Try to find matching mappings (can be multiple targets for one source)
            targets = flat_mapping.get(hf_key, [])

            if targets:
                for mlx_path, transform in targets:
                    # Apply transform if specified
                    tensor = hf_tensor
                    if transform:
                        tensor = transform(tensor)

                    # Build nested structure
                    WeightMapper._set_nested_value(mapped_weights, mlx_path, tensor)
                    mapped_count += 1
            else:
                # Weight not in mapping - might be intentionally skipped (e.g., lm_head)
                # or optional weight (e.g., conv_shortcut) - that's OK
                skipped_count += 1

        # Optional: uncomment for debugging
        # print(f"✅ Mapped {mapped_count} weights, skipped {skipped_count}")

        return mapped_weights

    @staticmethod
    def missing_required_targets(
        hf_weights: Dict[str, mx.array],
        mapping: List[WeightTarget],
        num_blocks: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> List[WeightTarget]:
        num_blocks, num_layers = WeightMapper._block_and_layer_counts(hf_weights, num_blocks, num_layers)
        return [
            target
            for target in mapping
            if WeightMapper._first_missing_name(hf_weights, target, num_blocks, num_layers) is not None
        ]

    @staticmethod
    def missing_required_names(
        hf_weights: Dict[str, mx.array],
        mapping: List[WeightTarget],
        num_blocks: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> List[str]:
        num_blocks, num_layers = WeightMapper._block_and_layer_counts(hf_weights, num_blocks, num_layers)
        names = (WeightMapper._first_missing_name(hf_weights, target, num_blocks, num_layers) for target in mapping)
        return [name for name in names if name is not None]

    @staticmethod
    def unmapped_names(
        hf_weights: Dict[str, mx.array],
        mapping: List[WeightTarget],
        num_blocks: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> List[str]:
        num_blocks, num_layers = WeightMapper._block_and_layer_counts(hf_weights, num_blocks, num_layers)
        used = WeightMapper._build_flat_mapping(mapping, num_blocks, num_layers)
        return sorted(name for name in hf_weights if name not in used)

    @staticmethod
    def _block_and_layer_counts(
        hf_weights: Dict[str, mx.array], num_blocks: Optional[int], num_layers: Optional[int]
    ) -> tuple[int, int]:
        if num_blocks is None:
            num_blocks = WeightMapper._detect_num_blocks(hf_weights)
        if num_layers is None:
            num_layers = WeightMapper._detect_num_layers(hf_weights)
        return num_blocks, num_layers

    @staticmethod
    def _first_missing_name(
        hf_weights: Dict[str, mx.array], target: WeightTarget, num_blocks: int, num_layers: int
    ) -> Optional[str]:
        # A required weight has to be in every block, layer and resnet the checkpoint has at its place in the pattern.
        # Only indices the checkpoint has count: _build_flat_mapping expands placeholders to detected or fixed counts
        # (every block pattern to the largest block count it detects, {i} to two where a VAE mid block has a single
        # attention), and trimmed checkpoints ship fewer blocks. Optional ones are left alone, since some sit in only
        # some blocks (the last up block of the FLUX.1 VAE has no upsampler), unless complete_when_present asks for the
        # per-block check on a family the checkpoint may leave out as a whole (a ControlNet's single blocks).
        if not target.required and not target.complete_when_present:
            return None
        found = any(name in hf_weights for name in WeightMapper._build_flat_mapping([target], num_blocks, num_layers))
        if target.required and not found:
            return WeightMapper._example_name(target)
        patterns = WeightMapper._indexed_patterns(target)
        for pattern in patterns:
            for indices in WeightMapper._present_indices(hf_weights, pattern, target.max_blocks):
                alternatives = [p for p in patterns if WeightMapper._placeholders(p) == set(indices)]
                names = [WeightMapper._fill(p, indices) for p in alternatives]
                if not any(name in hf_weights for name in names):
                    return names[0]
        return None

    @staticmethod
    def _indexed_patterns(target: WeightTarget) -> List[str]:
        # Source patterns whose every placeholder numbers the destination too. A source copied to every block
        # (one-to-many) keeps the any-name rule.
        indexed = []
        for pattern in target.from_pattern:
            placeholders = WeightMapper._placeholders(pattern)
            if placeholders and all(p in target.to_pattern for p in placeholders):
                if min(pattern.index(p) for p in placeholders) > 0:
                    indexed.append(pattern)
        return indexed

    @staticmethod
    def _present_indices(
        hf_weights: Dict[str, mx.array], pattern: str, max_blocks: Optional[int]
    ) -> List[Dict[str, int]]:
        # Every combination of indices the checkpoint has for this pattern, outermost placeholder first, so a
        # (block, res) pair counts only when that block has that resnet.
        placeholders = sorted(WeightMapper._placeholders(pattern), key=pattern.index)
        if not placeholders:
            return [{}]
        first = placeholders[0]
        prefix = pattern[: pattern.index(first)]
        found = set()
        for name in hf_weights:
            if name.startswith(prefix):
                head = name[len(prefix) :].split(".", 1)[0]
                if head.isdigit():
                    found.add(int(head))
        present = []
        for index in sorted(found):
            if first == "{block}" and max_blocks is not None and index >= max_blocks:
                continue
            inner = WeightMapper._present_indices(hf_weights, pattern.replace(first, str(index)), max_blocks)
            present.extend({first: index, **rest} for rest in inner)
        return present

    @staticmethod
    def _placeholders(pattern: str) -> set[str]:
        return {p for p in WeightMapper._PLACEHOLDERS if p in pattern}

    @staticmethod
    def _fill(pattern: str, indices: Dict[str, int]) -> str:
        for placeholder, index in indices.items():
            pattern = pattern.replace(placeholder, str(index))
        return pattern

    @staticmethod
    def _example_name(target: WeightTarget) -> str:
        name = target.from_pattern[0] if target.from_pattern else target.to_pattern
        for placeholder in WeightMapper._PLACEHOLDERS:
            name = name.replace(placeholder, "0")
        return name

    @staticmethod
    def _detect_num_blocks(hf_weights: Dict[str, mx.array]) -> int:
        block_numbers = set()
        for key in hf_weights.keys():
            # Match pattern: transformer_blocks.{number}.something
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))
                continue
            # Match pattern: single_transformer_blocks.{number}.something
            match = re.search(r"single_transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))

        if block_numbers:
            return max(block_numbers) + 1  # Blocks are 0-indexed
        return 0

    @staticmethod
    def _detect_num_layers(hf_weights: Dict[str, mx.array]) -> int:
        layer_numbers = set()
        for key in hf_weights.keys():
            # Match pattern: model.layers.{number}.something
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match:
                layer_numbers.add(int(match.group(1)))

        if layer_numbers:
            return max(layer_numbers) + 1  # Layers are 0-indexed
        return 28  # Default 28 layers for Qwen text encoder

    @staticmethod
    def _build_flat_mapping(
        mapping: List[WeightTarget], num_blocks: int = 0, num_layers: int = 28
    ) -> Dict[str, List[tuple[str, Optional[Callable[[mx.array], mx.array]]]]]:
        flat: Dict[str, List[tuple[str, Optional[Callable[[mx.array], mx.array]]]]] = {}

        def add_mapping(hf_key: str, mlx_path: str, transform: Optional[Callable[[mx.array], mx.array]]):
            if hf_key not in flat:
                flat[hf_key] = []
            flat[hf_key].append((mlx_path, transform))

        for target in mapping:
            # Expand placeholders for each pattern
            for hf_pattern in target.from_pattern:
                # Check which placeholders are present in BOTH patterns
                hf_has_block = "{block}" in hf_pattern
                to_has_block = "{block}" in target.to_pattern
                has_i = "{i}" in hf_pattern or "{i}" in target.to_pattern
                has_res = "{res}" in hf_pattern or "{res}" in target.to_pattern
                has_layer = "{layer}" in hf_pattern or "{layer}" in target.to_pattern

                # Handle multiple placeholders together
                if (hf_has_block or to_has_block) and has_res:
                    # Up blocks: expand both {block} and {res}
                    max_blocks = num_blocks if num_blocks > 0 else 4  # Default 4 for up_blocks
                    for block_num in range(max_blocks):
                        for res in range(3):  # 3 resnets per up_block
                            concrete_hf = hf_pattern.replace("{block}", str(block_num)).replace("{res}", str(res))
                            concrete_mlx = target.to_pattern.replace("{block}", str(block_num)).replace(
                                "{res}", str(res)
                            )
                            add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif hf_has_block and to_has_block:
                    # Both have {block} - standard one-to-one expansion
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    elif "visual.blocks" in hf_pattern or "visual.blocks" in target.to_pattern:
                        max_blocks = 32  # Visual blocks are always 32
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 4
                    for block_num in range(max_blocks):
                        concrete_hf = hf_pattern.replace("{block}", str(block_num))
                        concrete_mlx = target.to_pattern.replace("{block}", str(block_num))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif to_has_block and not hf_has_block:
                    # One-to-many: single HF key maps to multiple MLX targets (e.g., relative_attention_bias)
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 24  # Default for T5
                    for block_num in range(max_blocks):
                        concrete_mlx = target.to_pattern.replace("{block}", str(block_num))
                        add_mapping(hf_pattern, concrete_mlx, target.transform)
                elif has_layer:
                    # Expand {layer} for text encoder layers or visual blocks
                    max_layers = num_layers if num_layers > 0 else 28  # Default 28 for text encoder
                    for layer_num in range(max_layers):
                        concrete_hf = hf_pattern.replace("{layer}", str(layer_num))
                        concrete_mlx = target.to_pattern.replace("{layer}", str(layer_num))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif has_i:
                    # Expand {i} only (for mid_block resnets)
                    for i in range(2):  # 2 resnets in mid_block
                        concrete_hf = hf_pattern.replace("{i}", str(i))
                        concrete_mlx = target.to_pattern.replace("{i}", str(i))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif has_res:
                    # This shouldn't happen for VAE (encoder down_blocks are explicit)
                    # But handle it just in case
                    if "up_block" in hf_pattern:
                        for res in range(3):
                            concrete_hf = hf_pattern.replace("{res}", str(res))
                            concrete_mlx = target.to_pattern.replace("{res}", str(res))
                            add_mapping(concrete_hf, concrete_mlx, target.transform)
                else:
                    # No placeholder, use as-is
                    add_mapping(hf_pattern, target.to_pattern, target.transform)

        return flat

    @staticmethod
    def _set_nested_value(d: Dict, path: str, value: mx.array):
        parts = path.split(".")
        current = d
        i = 0

        while i < len(parts) - 1:
            part = parts[i]

            # Check if next part is a digit (list index)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                # This is a list, ensure it exists
                if part not in current:
                    current[part] = []
                # Ensure list is large enough
                idx = int(parts[i + 1])
                while len(current[part]) <= idx:
                    current[part].append({})
                current = current[part][idx]
                # Skip both the key and the index
                i += 2
            else:
                # Regular dict key
                if part not in current:
                    current[part] = {}
                current = current[part]
                i += 1

        # Set final value
        final_key = parts[-1]
        current[final_key] = value
