from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.flux.variants.depth.depth_util import DepthUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux1Depth(nn.Module):
    vae: VAE
    depth_pro: DepthPro
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
        model_config: ModelConfig = ModelConfig.dev_depth(),
    ):
        super().__init__()
        FluxInitializer.init_depth(
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
        depth_image_path: Path | str | None = None,
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
            depth_image_path=depth_image_path,
            num_inference_steps=num_inference_steps,
        )

        # 1. Create the initial latents
        latents = FluxLatentCreator.create_noise(
            seed=seed,
            height=config.height,
            width=config.width,
        )

        # 2. Encode the prompt
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.tokenizers["t5"],
            clip_tokenizer=self.tokenizers["clip"],
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # 3. Encode the depth map
        static_depth_map, depth_image = DepthUtil.encode_depth_map(
            vae=self.vae,
            depth_pro=self.depth_pro,
            config=config,
            image_path=config.image_path,
            depth_image_path=config.depth_image_path,
        )

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents, depth_image=depth_image)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 5.t Concatenate the updated latents with the (static) depth map
                hidden_states = mx.concatenate([latents, static_depth_map], axis=-1)

                # 6.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden_states,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # 7.t Take one denoise step
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

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
            depth_image_path=config.depth_image_path,
            generation_time=config.time_steps.format_dict["elapsed"],
        )
