from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping
from mflux.utils.version_util import VersionUtil


class Flux2BaseTrainingAdapter(TrainingAdapter):
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        quantize: int | None,
        model_factory: Callable[[ModelConfig, int | None], Any],
    ):
        self._model_config = model_config
        self._flux2 = model_factory(model_config, quantize)
        self._guidance: float = 1.0

    def model(self):
        return self._flux2

    def transformer(self):
        return self._flux2.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        self._guidance = training_spec.guidance
        return Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=training_spec.guidance,
            scheduler="flow_match_euler_discrete",
        )

    def freeze_base(self) -> None:
        self._flux2.vae.freeze()
        self._flux2.transformer.freeze()
        self._flux2.text_encoder.freeze()

    def _encode_output_latents(self, *, image_path: Path, width: int, height: int) -> tuple[mx.array, mx.array]:
        encoded = LatentCreator.encode_image(
            vae=self._flux2.vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=self._flux2.tiling_config,
        )
        encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
        encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
        encoded = Flux2LatentCreator.patchify_latents(encoded)
        encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=self._flux2.vae)

        clean_latents = Flux2LatentCreator.pack_latents(encoded)
        img_ids = Flux2LatentCreator.prepare_grid_ids(encoded, t_coord=0)
        return clean_latents, img_ids

    def _encode_prompt(self, *, prompt: str) -> tuple[mx.array, mx.array]:
        return Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self._flux2.tokenizers["qwen3"],
            text_encoder=self._flux2.text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:  # noqa: ARG002
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            module_path = target.module_path
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    module_path = target.module_path.format(block=b)
                    self._append_train_lora_weights(weights, self._flux2.transformer, module_path)
            else:
                self._append_train_lora_weights(weights, self._flux2.transformer, module_path)

        mx.save_safetensors(
            str(path),
            weights,
            metadata={
                "mflux_version": VersionUtil.get_mflux_version(),
                "model": training_spec.model,
            },
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Flux2LoRAMapping.get_mapping(),
            transformer=self._flux2.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
            bake_lora=False,
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Flux2LoRAMapping.get_mapping(),
            transformer=self._flux2.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
            bake_lora=False,
        )

    def _assistant_disabled(self):
        return TrainingUtil.assistant_disabled(self._flux2.transformer)

    @staticmethod
    def _append_train_lora_weights(weights: dict[str, mx.array], transformer, module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(transformer, module_path)

        weights[f"transformer.{module_path}.lora_A.weight"] = mx.transpose(train_lora.lora_A)
        weights[f"transformer.{module_path}.lora_B.weight"] = mx.transpose(train_lora.lora_B)
