from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Protocol

import mlx.core as mx


@dataclass
class LoRATarget:
    model_path: str
    possible_up_patterns: List[str]
    possible_down_patterns: List[str]
    possible_alpha_patterns: List[str] = field(default_factory=list)
    possible_lokr_w1_patterns: List[str] = field(default_factory=list)
    possible_lokr_w2_patterns: List[str] = field(default_factory=list)
    possible_dora_scale_patterns: List[str] = field(default_factory=list)
    up_transform: Callable[[mx.array], mx.array] | None = None
    down_transform: Callable[[mx.array], mx.array] | None = None
    lokr_w1_transform: Callable[[mx.array], mx.array] | None = None
    lokr_w2_transform: Callable[[mx.array], mx.array] | None = None


class LoRAMapping(Protocol):
    @staticmethod
    def get_mapping() -> List[LoRATarget]:
        return
