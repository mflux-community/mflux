from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.variants.z_image import ZImage
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
from mflux.utils.version_util import VersionUtil


class ZImageTrainingAdapter(TrainingAdapter):
    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        self._model_config = model_config
        self._z = ZImage(quantize=quantize, model_config=model_config, model_path=model_path)
        self._guidance: float | None = None

    def model(self):
        return self._z

    def transformer(self):
        return self._z.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        self._guidance = training_spec.guidance
        return Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=training_spec.guidance,
        )

    def freeze_base(self) -> None:
        # Mirror Flux.freeze behavior: freeze all base components.
        self._z.vae.freeze()
        self._z.transformer.freeze()
        self._z.text_encoder.freeze()

    def encode_data(
        self,
        *,
        data_id: int,
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,
    ) -> tuple[mx.array, Any]:
        encoded = LatentCreator.encode_image(vae=self._z.vae, image_path=image_path, height=height, width=width)
        clean_latents = ZImageLatentCreator.pack_latents(encoded, height=height, width=width)

        cap_feats = PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self._z.tokenizers["z_image"],
            text_encoder=self._z.text_encoder,
        )

        mx.eval(clean_latents, cap_feats)
        return clean_latents, cap_feats

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:  # noqa: ARG002
        return self._z.transformer(timestep=t, x=latents_t, cap_feats=cond, sigmas=sigmas)

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,
    ):
        # Show samples without the training adapter, matching inference behavior.
        with self._assistant_disabled():
            return self._z.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance,
            )

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:  # noqa: ARG002
        # Save in a Z-image-compatible namespace so it loads via ZImageLoRAMapping.
        weights: dict[str, mx.array] = {}
        # Only save the trainable LoRA (exclude assistant/training adapters).
        for target in training_spec.lora_layers.targets:
            module_path = target.module_path
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    module_path = target.module_path.format(block=b)
                    ZImageTrainingAdapter._append_train_lora_weights(weights, self._z.transformer, module_path)
            else:
                ZImageTrainingAdapter._append_train_lora_weights(weights, self._z.transformer, module_path)

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
            lora_mapping=ZImageLoRAMapping.get_mapping(),
            transformer=self._z.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
            bake_lora=False,
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=ZImageLoRAMapping.get_mapping(),
            transformer=self._z.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
            bake_lora=False,
        )

    @staticmethod
    def _append_train_lora_weights(weights: dict[str, mx.array], transformer, module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(transformer, module_path)

        weights[f"diffusion_model.{module_path}.lora_A.weight"] = mx.transpose(train_lora.lora_A)
        weights[f"diffusion_model.{module_path}.lora_B.weight"] = mx.transpose(train_lora.lora_B)

    def _assistant_disabled(self):
        # Context manager that temporarily disables the assistant/training adapter LoRA(s).
        return TrainingUtil.assistant_disabled(self._z.transformer)
