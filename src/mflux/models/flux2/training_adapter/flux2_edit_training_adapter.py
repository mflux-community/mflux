from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.training_adapter.flux2_base_training_adapter import Flux2BaseTrainingAdapter
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers


class Flux2EditTrainingAdapter(Flux2BaseTrainingAdapter):
    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        super().__init__(
            model_config=model_config,
            quantize=quantize,
            model_factory=lambda cfg, q: Flux2KleinEdit(model_config=cfg, quantize=q, model_path=model_path),
        )

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
        if input_image_path is None:
            raise ValueError("Edit training requires input_image for every example.")

        # Output image -> packed latents (training target)
        clean_latents, img_ids = self._encode_output_latents(image_path=image_path, width=width, height=height)

        # Prompt -> conditioning
        prompt_embeds, text_ids = self._encode_prompt(prompt=prompt)

        # Reference image -> conditioning (encoded at its own aspect-preserved size)
        image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
            vae=self._flux2.vae,
            tiling_config=self._flux2.tiling_config,
            image_paths=[input_image_path],
            batch_size=1,
        )
        if image_latents is None or image_latent_ids is None:
            raise ValueError("Edit training requires input_image conditioning.")

        mx.eval(clean_latents, prompt_embeds, text_ids, img_ids, image_latents, image_latent_ids)
        return (
            clean_latents,
            {
                "prompt_embeds": prompt_embeds,
                "text_ids": text_ids,
                "img_ids": img_ids,
                "image_latents": image_latents,
                "image_latent_ids": image_latent_ids,
            },
        )

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:  # noqa: ARG002
        hidden_states = mx.concatenate([latents_t, cond["image_latents"]], axis=1)
        img_ids = mx.concatenate([cond["img_ids"], cond["image_latent_ids"]], axis=1)
        noise = self._flux2.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=cond["prompt_embeds"],
            timestep=config.scheduler.timesteps[t],
            img_ids=img_ids,
            txt_ids=cond["text_ids"],
            guidance=None,
        )
        return noise[:, : latents_t.shape[1]]

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
        if not image_paths:
            raise ValueError("Edit training preview requires data/preview.*.")
        with self._assistant_disabled():
            image = self._flux2.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance,
                image_paths=image_paths,
            )
        return image.image
