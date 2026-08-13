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
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.flux.variants.concept_attention.attention_data import GenerationAttentionData
from mflux.models.flux.variants.concept_attention.concept_util import ConceptUtil
from mflux.models.flux.variants.concept_attention.transformer_concept import TransformerConcept
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux1ConceptFromImage(nn.Module):
    vae: VAE
    transformer: TransformerConcept
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        model_config: ModelConfig = ModelConfig.schnell(),
    ):
        super().__init__()
        FluxInitializer.init_concept(
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
        concept: str,
        image_path: Path | str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_strength: float | None = None,
        scheduler: str = "linear",
        heatmap_layer_indices: list[int] | None = None,
        heatmap_timesteps: list[int] | None = None,
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

        # 1. Create the initial latents from the reference image
        encoded_image = LatentCreator.encode_image(
            vae=self.vae,
            width=config.width,
            height=config.height,
            image_path=image_path,
            tiling_config=self.tiling_config,
        )

        # Create static noise for blending at each timestep
        static_noise = FluxLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
        )

        # Start with an appropriately noised version of the encoded image
        latents = LatentCreator.add_noise_by_interpolation(
            clean=FluxLatentCreator.pack_latents(latents=encoded_image, height=config.height, width=config.width),
            noise=static_noise,
            sigma=config.scheduler.sigmas[config.init_time_step],
        )

        # 2. Encode the main prompt
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.tokenizers["t5"],
            clip_tokenizer=self.tokenizers["clip"],
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # 3. Encode the concept prompt
        prompt_embeds_concept, pooled_prompt_embeds_concept = PromptEncoder.encode_prompt(
            prompt=concept,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.tokenizers["t5"],
            clip_tokenizer=self.tokenizers["clip"],
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        attention_data = GenerationAttentionData()
        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 5.t Predict the noise (we don't use the noise, only the attention)
                _, attention = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_concept=prompt_embeds_concept,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    pooled_prompt_embeds_concept=pooled_prompt_embeds_concept,
                )
                attention_data.append(attention)

                # 6.t Follow reverse diffusion trajectory
                latents = LatentCreator.add_noise_by_interpolation(
                    clean=FluxLatentCreator.pack_latents(
                        latents=encoded_image, height=config.height, width=config.width
                    ),
                    noise=static_noise,
                    sigma=config.scheduler.sigmas[t + 1],
                )

                # 7.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # Evaluate attention data to force MLX computation for progress tracking
                mx.eval(
                    [layer.img_attention for layer in attention.attention_information]
                    + [layer.concept_attention for layer in attention.attention_information]
                )

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 8. Call subscribers after loop
        ctx.after_loop(latents)

        # 9. Generate concept attention heatmap
        concept_heatmap = ConceptUtil.create_heatmap(
            concept=concept,
            attention_data=attention_data,
            height=config.height,
            width=config.width,
            layer_indices=heatmap_layer_indices or list(range(15, 19)),
            timesteps=heatmap_timesteps or list(range(config.num_inference_steps)),
        )

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
            image_path=image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
            concept_heatmap=concept_heatmap,
        )
