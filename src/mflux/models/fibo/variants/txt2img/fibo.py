from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.fibo.fibo_initializer import FIBOInitializer
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.model.fibo_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.fibo.model.fibo_text_encoder.smol_lm3_3b_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class FIBO(nn.Module):
    vae: Wan2_2_VAE
    transformer: FiboTransformer
    text_encoder: SmolLM3_3B_TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        model_config: ModelConfig = ModelConfig.fibo(),
    ):
        super().__init__()
        FIBOInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        # 0. Create a new config based on the model type and input parameters
        effective_guidance = guidance
        if "fibo-lite" in self.model_config.aliases:
            effective_guidance = 1.0  # distilled model, cond-only
        config = Config(
            width=width,
            height=height,
            guidance=effective_guidance,
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
                latent_creator=FiboLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                tiling_config=self.tiling_config,
            ),
        )

        # 2. Encode the prompt
        json_prompt, encoder_hidden_states, text_encoder_layers = PromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            tokenizer=self.tokenizers["fibo"],
            text_encoder=self.text_encoder,
            guidance=config.guidance,
        )

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=json_prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # 4.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    text_encoder_layers=text_encoder_layers,
                    encoder_hidden_states=encoder_hidden_states,
                )
                if config.guidance != 1.0:
                    noise = FIBO._apply_classifier_free_guidance(noise, config.guidance)

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
        latents = FiboLatentCreator.unpack_latents(latents, config.height, config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=json_prompt,
            quantization=self.bits,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    @staticmethod
    def _apply_classifier_free_guidance(noise: mx.array, guidance: float) -> mx.array:
        half = noise.shape[0] // 2
        noise_uncond = noise[:half]
        noise_text = noise[half:]
        return noise_uncond + guidance * (noise_text - noise_uncond)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=FIBOWeightDefinition,
        )
