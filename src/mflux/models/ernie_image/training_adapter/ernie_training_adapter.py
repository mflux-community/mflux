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
from mflux.models.ernie_image.latent_creator.ernie_latent_creator import ErnieLatentCreator
from mflux.models.ernie_image.model.ernie_text_encoder.prompt_encoder import ErniePromptEncoder
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.models.ernie_image.weights.ernie_lora_mapping import ErnieLoRAMapping
from mflux.utils.version_util import VersionUtil


class ErnieTrainingAdapter(TrainingAdapter):
    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        self._model_config = model_config
        self._ernie = ErnieImage(quantize=quantize, model_config=model_config, model_path=model_path)
        self._guidance: float | None = None

    def model(self):
        return self._ernie

    def transformer(self):
        return self._ernie.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        steps = self._model_config.lora_training_steps or training_spec.steps
        guidance = self._model_config.lora_training_guidance
        if guidance is None:
            guidance = training_spec.guidance
        self._guidance = guidance
        return Config(
            model_config=self._model_config,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance=guidance,
            scheduler="linear",
        )

    def freeze_base(self) -> None:
        self._ernie.vae.freeze()
        self._ernie.transformer.freeze()
        self._ernie.text_encoder.freeze()

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
        encoded = LatentCreator.encode_image(vae=self._ernie.vae, image_path=image_path, height=height, width=width)
        clean_latents = ErnieLatentCreator.pack_latents(encoded, height=height, width=width)
        clean_latents = ErnieLatentCreator.bn_normalize_latents(clean_latents, vae=self._ernie.vae)
        text_bth, text_lens = ErniePromptEncoder.build_text_batch(
            prompts=[prompt],
            tokenizer=self._ernie.tokenizers["ernie"],
            text_encoder=self._ernie.text_encoder,
        )
        mx.eval(clean_latents, text_bth, text_lens)
        return clean_latents, {"text_bth": text_bth, "text_lens": text_lens}

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:
        text_bth, text_lens = cond["text_bth"], cond["text_lens"]
        timestep = sigmas[t].reshape((1,)) * 1000
        return self._ernie.transformer(
            hidden_states=latents_t,
            timestep=mx.broadcast_to(timestep, (1,)),
            text_bth=text_bth,
            text_lens=text_lens,
        )

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
        canonical_steps = self._model_config.lora_training_steps or steps
        image = self._ernie.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=canonical_steps,
            height=height,
            width=width,
            guidance=self._guidance,
        )
        self._ernie.prompt_cache = {}
        return image

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            module_path = target.module_path
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    mp = module_path.format(block=b)
                    ErnieTrainingAdapter._append_lora_weights(weights, self._ernie.transformer, mp)
            else:
                ErnieTrainingAdapter._append_lora_weights(weights, self._ernie.transformer, module_path)
        mx.save_safetensors(
            str(path),
            weights,
            metadata={"mflux_version": VersionUtil.get_mflux_version(), "model": training_spec.model},
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=ErnieLoRAMapping.get_mapping(),
            transformer=self._ernie.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=ErnieLoRAMapping.get_mapping(),
            transformer=self._ernie.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
        )

    @staticmethod
    def _append_lora_weights(weights: dict, transformer, module_path: str) -> None:
        lora = TrainingUtil.get_train_lora(transformer, module_path)
        weights[f"transformer.{module_path}.lora_A.weight"] = mx.transpose(lora.lora_A)
        weights[f"transformer.{module_path}.lora_B.weight"] = mx.transpose(lora.lora_B)
