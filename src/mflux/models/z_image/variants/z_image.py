from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil


class ZImage(nn.Module):
    vae: VAE
    text_encoder: TextEncoder
    transformer: ZImageTransformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        model_config: ModelConfig = ModelConfig.z_image_turbo(),
    ):
        super().__init__()
        ZImageInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        supports_guidance = bool(self.model_config.supports_guidance)
        if not supports_guidance:
            guidance = 0.0

        if scheduler is None:
            scheduler = "flow_match_euler_discrete" if supports_guidance else "linear"

        # 0. Create a new config based on the model type and input parameters
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=image_strength,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )
        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=ZImageLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                tiling_config=self.tiling_config,
            ),
        )
        text_encodings, negative_encodings = self._encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=config.guidance,
        )

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)

        for t in config.time_steps:
            try:
                # 4.t Predict the noise
                sigma_t = config.scheduler.sigmas[t].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t
                noise = predict(
                    latents=latents,
                    timestep=timestep,
                    sigmas=config.scheduler.sigmas,
                    text_encodings=text_encodings,
                    negative_encodings=negative_encodings,
                    guidance=config.guidance,
                )

                # 5.t Take one denoise step
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

                # 6.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 7. Call subscribers after loop
        ctx.after_loop(latents)

        # 8. Decode the latents and return the image
        decoded = self._decode_latents(latents=latents, config=config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
        )

    def _encode_prompts(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array | None]:
        text_encodings = PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
        )
        if guidance <= 1.0:
            return text_encodings, None
        negative_text = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        negative_encodings = PromptEncoder.encode_prompt(
            prompt=negative_text,
            tokenizer=self.tokenizers["z_image"],
            text_encoder=self.text_encoder,
        )
        return text_encodings, negative_encodings

    def _decode_latents(self, *, latents: mx.array, config: Config) -> mx.array:
        unpacked = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
        return VAEUtil.decode(vae=self.vae, latent=unpacked, tiling_config=self.tiling_config)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ZImageWeightDefinition,
        )

    @staticmethod
    def _predict(transformer: ZImageTransformer):
        def predict(
            latents: mx.array,
            timestep: mx.array,
            sigmas: mx.array,
            text_encodings: mx.array,
            negative_encodings: mx.array | None,
            guidance: float,
        ) -> mx.array:
            noise = transformer(
                timestep=timestep,
                x=latents,
                cap_feats=text_encodings,
                sigmas=sigmas,
            )
            if negative_encodings is None:
                return noise
            negative_noise = transformer(
                timestep=timestep,
                x=latents,
                cap_feats=negative_encodings,
                sigmas=sigmas,
            )
            return noise + guidance * (noise - negative_noise)

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)
