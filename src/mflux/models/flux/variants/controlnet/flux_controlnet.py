import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.flux.variants.controlnet.controlnet_util import ControlnetUtil
from mflux.models.flux.variants.controlnet.transformer_controlnet import TransformerControlnet
from mflux.models.flux.weights.flux_weight_definition import FluxControlnetWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil, StrOrBytesPath
from mflux.utils.metadata_reader import MetadataReader


class Flux1Controlnet(nn.Module):
    vae: VAE
    transformer: Transformer
    transformer_controlnet: TransformerControlnet
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        controlnet_path: str | None = None,
        model_config: ModelConfig = ModelConfig.dev_controlnet_canny(),
    ):
        super().__init__()
        FluxInitializer.init_controlnet(
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
        controlnet_image_path: StrOrBytesPath,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        controlnet_strength: float = 1.0,
        scheduler: str = "linear",
    ) -> GeneratedImage:
        # 0. Create a new config based on the model type and input parameters
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            controlnet_strength=controlnet_strength,
        )

        # 1. Encode the controlnet reference image
        controlnet_condition, canny_image = ControlnetUtil.encode_image(
            vae=self.vae,
            width=config.width,
            height=config.height,
            controlnet_image_path=controlnet_image_path,
            is_canny=self.model_config.is_canny(),
        )

        # 2. Create the initial latents
        latents = FluxLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
        )

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
        ctx.before_loop(latents, canny_image=canny_image)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 5.t Compute controlnet samples
                controlnet_block_samples, controlnet_single_block_samples = self.transformer_controlnet(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_condition=controlnet_condition,
                )

                # 6.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
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

        # 11. Read metadata from the controlnet image if available
        init_metadata = MetadataReader.read_all_metadata(controlnet_image_path) if controlnet_image_path else None

        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            controlnet_image_path=controlnet_image_path,
            generation_time=config.time_steps.format_dict["elapsed"],
            init_metadata=init_metadata,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=FluxControlnetWeightDefinition,
        )
