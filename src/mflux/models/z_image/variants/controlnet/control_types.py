from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ControlType(str, Enum):
    pose = "pose"
    canny = "canny"
    hed = "hed"
    depth = "depth"
    mlsd = "mlsd"


@dataclass(frozen=True)
class ControlSpec:
    type: ControlType
    image_path: Path | str
    strength: float = 1.0
