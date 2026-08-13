from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux1InContextDev(nn.Module):
    vae: VAE
    transformer: Transformer
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        model_config: ModelConfig = ModelConfig.dev(),
    ):
        super().__init__()
        FluxInitializer.init(
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
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
    ) -> GeneratedImage:
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

        # 1. Encode the reference image
        encoded_image = LatentCreator.encode_image(
            vae=self.vae,
            width=config.width,
            height=config.height,
            image_path=config.image_path,
            tiling_config=self.tiling_config,
        )

        # 2. Create the initial latents and keep the initial static noise for later blending
        static_noise = Flux1InContextDev._create_in_context_latents(seed=seed, config=config)
        latents = mx.array(static_noise)

        # 3. Encode the prompt
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.tokenizers["t5"],
            clip_tokenizer=self.tokenizers["clip"],
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 5.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # 6.t Take one denoise step and update latents
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

                # 7.t Override the left-hand side of latents by linearly interpolating between latents and static noise
                latents = Flux1InContextDev._update_latents(
                    t=t,
                    config=config,
                    latents=latents,
                    encoded_image=encoded_image,
                    static_noise=static_noise,
                    sigmas=config.scheduler.sigmas,
                )

                # 8.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 9. Call subscribers after loop
        ctx.after_loop(latents)

        # 10. Decode the latent array and return the image
        latents = FluxLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
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
        )

    @staticmethod
    def _create_in_context_latents(seed: int, config: Config):
        # 1. Double the width for side-by-side generation
        config.width = 2 * config.width

        # 2. Create the initial latents with doubled width
        latent_height = config.height // 8
        latent_width = config.width // 8

        # 3. Create noise with appropriate dimensions
        static_noise = mx.random.normal(shape=[1, 16, latent_height, latent_width], key=mx.random.key(seed))
        latents = FluxLatentCreator.pack_latents(latents=static_noise, height=config.height, width=config.width)
        return latents

    @staticmethod
    def _update_latents(
        t: int,
        config: Config,
        latents: mx.array,
        encoded_image: mx.array,
        static_noise: mx.array,
        sigmas: mx.array,
    ) -> mx.array:
        # 1. Unpack the latents
        unpacked = FluxLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        unpacked_static_noise = FluxLatentCreator.unpack_latents(
            latents=static_noise, height=config.height, width=config.width
        )

        # 2. Calculate latent_width from the config (original width is half of current width)
        latent_width = (config.width // 2) // 8

        # 3. Override the left side with the reference image blended with appropriate noise for current timestep
        unpacked[:, :, :, 0:latent_width] = LatentCreator.add_noise_by_interpolation(
            clean=encoded_image[:, :, :, 0:latent_width],
            noise=unpacked_static_noise[:, :, :, 0:latent_width],
            sigma=sigmas[t + 1],
        )

        # 4. Repack the latents
        return FluxLatentCreator.pack_latents(latents=unpacked, height=config.height, width=config.width)
