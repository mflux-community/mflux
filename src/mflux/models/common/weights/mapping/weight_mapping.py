from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

import mlx.core as mx


@dataclass
class WeightTarget:
    to_pattern: str
    from_pattern: List[str]
    transform: Optional[Callable[[mx.array], mx.array]] = None
    required: bool = True
    max_blocks: Optional[int] = None
    complete_when_present: bool = False


class WeightMapping(Protocol):
    @staticmethod
    def get_mapping() -> List[WeightTarget]:
        return []
