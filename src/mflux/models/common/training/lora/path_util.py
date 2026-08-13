from __future__ import annotations

from typing import Any

from mflux.models.common.training.state.training_spec import BlockRange, LoraTargetSpec


def _iter_parts(path: str) -> list[str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ValueError("module_path cannot be empty")
    return parts


def get_at_path(root: Any, path: str) -> Any:
    current = root
    for part in _iter_parts(path):
        if part.isdigit():
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def set_at_path(root: Any, path: str, value: Any) -> None:
    parts = _iter_parts(path)
    if len(parts) == 1:
        parent = root
        final = parts[0]
    else:
        parent = get_at_path(root, ".".join(parts[:-1]))
        final = parts[-1]

    if final.isdigit():
        parent[int(final)] = value
    elif isinstance(parent, dict):
        parent[final] = value
    else:
        setattr(parent, final, value)


def expand_module_paths(target: LoraTargetSpec) -> list[str]:
    if target.blocks is None:
        return [target.module_path]
    if "{block}" not in target.module_path:
        raise ValueError(f"Target has blocks specified but module_path contains no '{{block}}': {target.module_path}")
    blocks: BlockRange = target.blocks
    return [target.module_path.format(block=b) for b in blocks.get_blocks()]


def expand_module_paths_from_targets(targets: list[LoraTargetSpec]) -> list[tuple[str, int, float | None]]:
    expanded: list[tuple[str, int, float | None]] = []
    for t in targets:
        expanded.extend((p, t.rank, t.alpha) for p in expand_module_paths(t))
    return expanded
