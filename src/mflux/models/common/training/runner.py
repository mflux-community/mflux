from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import mlx.core.random as mx_random  # type: ignore[import-not-found]
from PIL import Image as PILImage

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.dataset.batch import DataItem
from mflux.models.common.training.dataset.data_cache import TrainingDataCache
from mflux.models.common.training.dataset.dataset import Dataset
from mflux.models.common.training.dataset.disk_backed_data import DiskBackedData
from mflux.models.common.training.dataset.iterator import Iterator
from mflux.models.common.training.lora.target_injector import inject_lora_targets
from mflux.models.common.training.optimization.optimizer import Optimizer
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.training_state import TrainingState
from mflux.models.common.training.state.zip_util import ZipUtil
from mflux.models.common.training.statistics.statistics import Statistics
from mflux.models.common.training.trainer import TrainingTrainer
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.ernie_image.training_adapter.ernie_training_adapter import ErnieTrainingAdapter
from mflux.models.flux2.training_adapter.flux2_edit_training_adapter import Flux2EditTrainingAdapter
from mflux.models.flux2.training_adapter.flux2_training_adapter import Flux2TrainingAdapter
from mflux.models.z_image.training_adapter.z_image_training_adapter import ZImageTrainingAdapter
from mflux.utils.exceptions import StopTrainingException


class TrainingRunner:
    # Z-Image-Turbo training benefits from a training adapter (assistant LoRA).
    ZIMAGE_TURBO_TRAINING_ADAPTER = "ostris/zimage_turbo_training_adapter:zimage_turbo_training_adapter_v2.safetensors"

    @staticmethod
    def _disable_assistant_loras(transformer) -> None:
        # Ensure any assistant LoRA(s) do not affect inference outputs after training.
        for lora in TrainingUtil.iter_assistant_loras(transformer):
            lora.scale = 0.0

    @staticmethod
    def _resolve_data_dimensions(*, training_spec: TrainingSpec, image_path) -> tuple[int, int]:
        # Infer per-image size from the image file (may vary per image).
        with PILImage.open(image_path.resolve()) as img:
            width, height = img.size
        return TrainingUtil.resolve_dimensions(
            width=width,
            height=height,
            max_resolution=training_spec.max_resolution,
            default_max_resolution=None,
            error_template=f"Image too small for training (needs >=16px): {image_path} ({{width}}x{{height}})",
        )

    @staticmethod
    def train(*, config_path: str | None, resume_path: str | None) -> tuple[TrainingAdapter, TrainingSpec]:
        training_spec = TrainingSpec.resolve(config_path=config_path, resume_path=resume_path)

        # Bound MLX's buffer-cache pool for the whole run (encoding + train loop). Freed
        # buffers above the cap return to the OS instead of being retained for reuse.
        if training_spec.cache_limit_gb:
            mx.set_cache_limit(int(training_spec.cache_limit_gb * 1e9))

        # Set global seed for MLX randomness
        mx_random.seed(training_spec.seed)

        model_config = ModelConfig.from_name(training_spec.model)
        is_zimage_turbo = model_config.model_name == ModelConfig.z_image_turbo().model_name
        is_zimage = model_config.model_name in {
            ModelConfig.z_image().model_name,
            ModelConfig.z_image_turbo().model_name,
        }
        is_ernie = model_config.model_name in {
            ModelConfig.ernie_image_turbo().model_name,
            ModelConfig.ernie_image().model_name,
        }
        is_flux2 = model_config.model_name.startswith("black-forest-labs/FLUX.2")
        is_flux2_base = model_config.model_name.startswith("black-forest-labs/FLUX.2-klein-base")
        if training_spec.is_edit and not is_flux2_base:
            raise ValueError("Edit training currently supports only FLUX.2-klein-base models.")
        if is_ernie:
            adapter = ErnieTrainingAdapter(
                model_config=model_config, quantize=training_spec.quantize, model_path=training_spec.model_path
            )
        elif is_zimage:
            adapter = ZImageTrainingAdapter(
                model_config=model_config, quantize=training_spec.quantize, model_path=training_spec.model_path
            )
        elif training_spec.is_edit:
            adapter = Flux2EditTrainingAdapter(
                model_config=model_config, quantize=training_spec.quantize, model_path=training_spec.model_path
            )
        elif is_flux2:
            adapter = Flux2TrainingAdapter(
                model_config=model_config, quantize=training_spec.quantize, model_path=training_spec.model_path
            )
        else:
            raise ValueError("Flux1 training is no longer supported.")

        if training_spec.low_ram:
            model = adapter.model()
            if hasattr(model, "tiling_config") and model.tiling_config is None:
                model.tiling_config = TilingConfig()

        # For Z-Image-Turbo we always apply the assistant training adapter (automatic, no config needed).
        if is_zimage_turbo:
            adapter.load_training_adapter(
                path=TrainingRunner.ZIMAGE_TURBO_TRAINING_ADAPTER,
                scale=1.0,
            )

        # Apply LoRA layers either by loading a saved adapter (resume) or by injecting fresh layers
        if training_spec.lora_layers.state_path is not None:
            ZipUtil.unzip(
                zip_path=training_spec.checkpoint_path,
                filename=training_spec.lora_layers.state_path,
                loader=lambda lora_file: adapter.load_lora_adapter(path=lora_file),
            )
        else:
            inject_lora_targets(adapter.transformer(), training_spec.lora_layers.targets)

        dataset = None
        iterator = None
        optimizer = None
        statistics = None
        training_state = None

        # Prepare dataset data
        if training_spec.low_ram:
            if training_spec.data_root is None:
                raise ValueError("low_ram requires TrainingSpec.data_root to be set")
            cache_paths = TrainingDataCache.wipe_and_init(data_root=Path(training_spec.data_root))
            for i, item in enumerate(training_spec.data):
                width, height = TrainingRunner._resolve_data_dimensions(
                    training_spec=training_spec, image_path=item.image
                )
                clean_latents, cond = adapter.encode_data(
                    data_id=i,
                    image_path=item.image,
                    prompt=item.prompt,
                    width=width,
                    height=height,
                    input_image_path=item.input_image,
                )
                TrainingDataCache.save_item(
                    paths=cache_paths,
                    data_id=i,
                    prompt=item.prompt,
                    image_path=item.image,
                    width=width,
                    height=height,
                    clean_latents=clean_latents,
                    cond=cond,
                )
                del clean_latents, cond
                gc.collect()
                mx.clear_cache()

            dataset = Dataset(DiskBackedData(data_specs=training_spec.data, cache_paths=cache_paths))
        else:
            # Encode dataset data (deterministic, upfront)
            encoded_data: list[DataItem] = []
            for i, item in enumerate(training_spec.data):
                width, height = TrainingRunner._resolve_data_dimensions(
                    training_spec=training_spec, image_path=item.image
                )
                clean_latents, cond = adapter.encode_data(
                    data_id=i,
                    image_path=item.image,
                    prompt=item.prompt,
                    width=width,
                    height=height,
                    input_image_path=item.input_image,
                )
                encoded_data.append(
                    DataItem(
                        data_id=i,
                        prompt=item.prompt,
                        image_path=item.image,
                        clean_latents=clean_latents,
                        cond=cond,
                        width=width,
                        height=height,
                    )
                )
            dataset = Dataset(encoded_data)
        iterator = Iterator.from_spec(training_spec=training_spec, dataset=dataset)
        optimizer = Optimizer.from_spec(training_spec)
        statistics = Statistics.from_spec(training_spec)
        training_state = TrainingState(iterator=iterator, optimizer=optimizer, statistics=statistics)

        try:
            TrainingTrainer.train(adapter=adapter, training_spec=training_spec, training_state=training_state)
            if is_zimage_turbo:
                TrainingRunner._disable_assistant_loras(adapter.transformer())
            return adapter, training_spec
        except StopTrainingException:
            training_state.save(adapter, training_spec)
            raise
        finally:
            # Best-effort in-memory cleanup; disk outputs remain untouched.
            if training_state is not None:
                del training_state
            if iterator is not None:
                del iterator
            if optimizer is not None:
                del optimizer
            if statistics is not None:
                del statistics
            if dataset is not None:
                del dataset
            gc.collect()
            mx.clear_cache()
