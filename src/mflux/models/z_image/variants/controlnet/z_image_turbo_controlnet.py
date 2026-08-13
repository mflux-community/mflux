from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.variants.controlnet.control_types import ControlSpec
from mflux.models.z_image.variants.controlnet.controlnet_util import ZImageControlnetUtil
from mflux.models.z_image.variants.controlnet.transformer_controlnet import ZImageControlNet
from mflux.models.z_image.weights.z_image_controlnet_weight_definition import ZImageControlnetWeightDefinition
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil


class ZImageTurboControlnet(nn.Module):
    vae: VAE
    text_encoder: TextEncoder
    transformer: ZImageTransformer
    controlnet: ZImageControlNet

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.z_image_turbo_controlnet_union_2_1(),
    ):
        super().__init__()
        ZImageInitializer.init_controlnet(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        controls: list[ControlSpec],
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        controlnet_strength: float = 0.8,
        scheduler: str = "linear",
    ) -> Image.Image:
        config = Config(
            width=width,
            height=height,
            guidance=0.0,  # Turbo uses no guidance
            scheduler=scheduler,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            controlnet_strength=controlnet_strength,
        )

        # 1) Latents
        latents = ZImageLatentCreator.create_noise(seed=seed, height=config.height, width=config.width)

        # 2) Prompt encoding
        text_encodings = PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
        )

        # 3) Control encodings (1+)
        encoded_controls = ZImageControlnetUtil.encode_controls(
            vae=self.vae,
            width=config.width,
            height=config.height,
            controls=controls,
            control_in_dim=self.controlnet.config.control_in_dim,
        )

        # 4) Callbacks
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents, control_images=encoded_controls.images)

        for t in config.time_steps:
            try:
                # Per-control residuals; sum into a single dict for this step
                merged: dict[int, mx.array] = {}
                for control_latent, control_scale in zip(encoded_controls.control_latents, encoded_controls.strengths):
                    samples = self.controlnet(
                        x=latents,
                        t=t,
                        sigmas=config.scheduler.sigmas,
                        cap_feats=text_encodings,
                        control_context=control_latent,
                        conditioning_scale=(
                            1.0 if config.controlnet_strength is None else float(config.controlnet_strength)
                        )
                        * float(control_scale),
                    )
                    for k, v in samples.items():
                        merged[k] = merged[k] + v if k in merged else v

                # Predict noise with injected control residuals
                noise = self.transformer(
                    timestep=t,
                    x=latents,
                    cap_feats=text_encodings,
                    sigmas=config.scheduler.sigmas,
                    controlnet_block_samples=merged,
                )

                # Denoise step
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # Decode
        latents = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)

        # Note: `GeneratedImage` metadata currently has a single `controlnet_image_path` field; we store the first one.
        first_control_path = None
        if controls and isinstance(controls[0].image_path, (str, Path)):
            first_control_path = str(controls[0].image_path)

        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            controlnet_image_path=first_control_path,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def save_model(self, base_path: str) -> None:
        from mflux.models.common.weights.saving.model_saver import ModelSaver

        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ZImageControlnetWeightDefinition,
        )
