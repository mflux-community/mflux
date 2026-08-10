from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import Tokenizer
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.flux.model.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.flux.model.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer
from mflux.models.flux.variants.redux.redux_util import ReduxUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux1Redux(nn.Module):
    vae: VAE
    image_encoder: SiglipVisionTransformer
    image_embedder: ReduxEncoder
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
        FluxInitializer.init_redux(
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
        redux_image_paths: list[Path | str],
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        redux_image_strengths: list[float] | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
    ) -> GeneratedImage:
        # 0. Create a new config based on the model type and input parameters
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_strength=image_strength,
            model_config=self.model_config,
            redux_image_paths=redux_image_paths,
            num_inference_steps=num_inference_steps,
            redux_image_strengths=redux_image_strengths,
        )

        # 1. Create the initial latents
        latents = FluxLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
        )

        # 2. Get prompt embeddings by fusing the prompt and image embeddings
        prompt_embeds, pooled_prompt_embeds = Flux1Redux._get_prompt_embeddings(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.tokenizers["t5"],
            clip_tokenizer=self.tokenizers["clip"],
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
            image_paths=config.redux_image_paths,
            image_encoder=self.image_encoder,
            image_embedder=self.image_embedder,
            image_strengths=config.redux_image_strengths,
        )

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 4.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
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

        # 8. Decode the latent array and return the image
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
            redux_image_paths=config.redux_image_paths,
            redux_image_strengths=config.redux_image_strengths,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    @staticmethod
    def _get_prompt_embeddings(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        t5_tokenizer: Tokenizer,
        clip_tokenizer: Tokenizer,
        t5_text_encoder: T5Encoder,
        clip_text_encoder: CLIPEncoder,
        image_paths: list[str] | list[Path],
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
        image_strengths: list[float] | None = None,
    ) -> tuple[mx.array, mx.array]:
        # 1. Encode the prompt
        prompt_embeds_txt, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=prompt_cache,
            t5_tokenizer=t5_tokenizer,
            clip_tokenizer=clip_tokenizer,
            t5_text_encoder=t5_text_encoder,
            clip_text_encoder=clip_text_encoder,
        )

        # 2. Encode the image(s) using the Siglip and Redux encoder
        image_embeds = ReduxUtil.embed_images(
            image_paths=image_paths,
            image_encoder=image_encoder,
            image_embedder=image_embedder,
            image_strengths=image_strengths,
        )

        # 3. Join text embeddings with all image embeddings
        prompt_embeds = mx.concatenate([prompt_embeds_txt] + image_embeds, axis=1)

        return prompt_embeds, pooled_prompt_embeds
