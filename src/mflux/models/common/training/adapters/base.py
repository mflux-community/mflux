from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.training.state.training_spec import TrainingSpec


class TrainingAdapter(Protocol):
    def model(self) -> nn.Module: ...

    def transformer(self) -> nn.Module: ...

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config: ...

    def freeze_base(self) -> None: ...

    def set_gradient_checkpointing(self, enabled: bool) -> None: ...

    def encode_data(
        self,
        *,
        data_id: int,
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,
    ) -> tuple[mx.array, Any]: ...

    def predict_noise(
        self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config
    ) -> mx.array: ...

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,
    ) -> Image.Image: ...

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None: ...

    def load_lora_adapter(self, *, path: str | Path) -> None: ...

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None: ...

    def sample_sigma(self, *, rng: random.Random, image_seq_len: int) -> float | None:  # noqa: ARG002
        # The noise level of a training step, for an adapter whose reference trainer draws it from a continuous
        # distribution (FIBO 2). None, the default, draws an index into the scheduler's grid.
        return None

    def release_text_encoder(self, training_spec: TrainingSpec) -> bool:  # noqa: ARG002
        # Called once the dataset is encoded. An adapter that no longer needs its text encoder frees it and returns True.
        return False
