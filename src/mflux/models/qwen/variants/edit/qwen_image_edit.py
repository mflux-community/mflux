import math
from pathlib import Path

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class QwenImageEdit(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        model_config: ModelConfig = ModelConfig.qwen_image_edit(),
    ):
        super().__init__()
        QwenImageInitializer.init_edit(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            model_config=model_config,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=QwenWeightDefinition,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: list[str],
        num_inference_steps: int = 4,
        height: int | None = None,
        width: int | None = None,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        scheduler: str = "linear",
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        config, vl_width, vl_height, vae_width, vae_height = self._compute_dimensions(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_paths=image_paths,
            num_inference_steps=num_inference_steps,
        )
        timesteps = config.scheduler.timesteps
        time_steps = tqdm(range(len(timesteps)))

        # 1. Create initial latents
        latents = QwenLatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
        )

        # 2. Encode the prompt
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self._encode_prompts_with_images(
            prompt=prompt,
            config=config,
            vl_width=vl_width,
            vl_height=vl_height,
            image_paths=image_paths,
            negative_prompt=negative_prompt,
        )

        # 3. Generate image conditioning latents
        static_image_latents, qwen_image_ids, cond_h_patches, cond_w_patches, num_images = (
            QwenEditUtil.create_image_conditioning_latents(
                vae=self.vae,
                width=vae_width,
                height=vae_height,
                vl_width=vl_width,
                vl_height=vl_height,
                image_paths=image_paths,
                tiling_config=self.tiling_config,
            )
        )

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in time_steps:
            try:
                # 5.t Concatenate the updated latents with the static image latents
                hidden_states = mx.concatenate([latents, static_image_latents], axis=1)
                hidden_states_neg = mx.concatenate([latents, static_image_latents], axis=1)

                # 6.t Predict the noise
                if num_images > 1:
                    cond_image_grid = [(1, cond_h_patches, cond_w_patches) for _ in range(num_images)]
                else:
                    cond_image_grid = (1, cond_h_patches, cond_w_patches)

                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=cond_image_grid,
                )[:, : latents.shape[1]]
                noise_negative = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden_states_neg,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_mask=negative_prompt_mask,
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=cond_image_grid,
                )[:, : latents.shape[1]]
                guided_noise = QwenImage.compute_guided_noise(noise, noise_negative, config.guidance)

                # 7.t Take one denoise step
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)

                # 8.t Call subscribers in-loop
                ctx.in_loop(t, latents, time_steps=time_steps)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents, time_steps=time_steps)
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(timesteps)}")

        # 9. Call subscribers after loop
        ctx.after_loop(latents)

        # 10. Decode the latent array and return the image
        latents = QwenLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
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
            image_paths=image_paths,
            generation_time=time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
        )

    def _encode_prompts_with_images(
        self,
        prompt: str,
        negative_prompt: str,
        image_paths: list[str],
        config,
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        tokenizer = self.tokenizers["qwen_vl"]
        pos_input_ids, pos_attention_mask, pos_pixel_values, pos_image_grid_thw = tokenizer.tokenize_with_image(
            prompt, image_paths, vl_width=vl_width, vl_height=vl_height
        )

        pos_hidden_states = self.qwen_vl_encoder(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            pixel_values=pos_pixel_values,
            image_grid_thw=pos_image_grid_thw,
        )

        neg_prompt = negative_prompt if negative_prompt is not None else ""

        # Use real MLX tokenizer for negative prompt
        neg_input_ids, neg_attention_mask, neg_pixel_values, neg_image_grid_thw = tokenizer.tokenize_with_image(
            neg_prompt, image_paths, vl_width=vl_width, vl_height=vl_height
        )

        neg_hidden_states = self.qwen_vl_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            pixel_values=neg_pixel_values,
            image_grid_thw=neg_image_grid_thw,
        )

        final_prompt_embeds = pos_hidden_states[0].astype(mx.float16)
        final_prompt_mask = pos_hidden_states[1].astype(mx.float16)

        return (
            final_prompt_embeds,  # prompt_embeds
            final_prompt_mask,  # prompt_mask
            neg_hidden_states[0].astype(mx.float16),  # negative_prompt_embeds
            neg_hidden_states[1].astype(mx.float16),  # negative_prompt_mask
        )

    def _compute_dimensions(
        self,
        image_paths: list[str],
        num_inference_steps: int,
        height: int | None,
        width: int | None,
        guidance: float,
        image_path: Path | str | None,
        scheduler: str,
    ) -> tuple[Config, int, int, int, int]:
        last_image = ImageUtil.load_image(image_paths[-1]).convert("RGB")
        image_size = last_image.size

        target_area = 1024 * 1024
        ratio = image_size[0] / image_size[1]
        calculated_width = math.sqrt(target_area * ratio)
        calculated_height = calculated_width / ratio
        calculated_width = round(calculated_width / 32) * 32
        calculated_height = round(calculated_height / 32) * 32

        use_height = height or int(calculated_height)
        use_width = width or int(calculated_width)

        vae_scale_factor = 8
        multiple_of = vae_scale_factor * 2
        use_width = use_width // multiple_of * multiple_of
        use_height = use_height // multiple_of * multiple_of

        config = Config(
            width=use_width,
            height=use_height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        CONDITION_IMAGE_SIZE = 384 * 384
        condition_ratio = image_size[0] / image_size[1]
        vl_width = math.sqrt(CONDITION_IMAGE_SIZE * condition_ratio)
        vl_height = vl_width / condition_ratio
        vl_width = round(vl_width / 32) * 32
        vl_height = round(vl_height / 32) * 32

        VAE_IMAGE_SIZE = 1024 * 1024
        vae_ratio = image_size[0] / image_size[1]
        vae_width = math.sqrt(VAE_IMAGE_SIZE * vae_ratio)
        vae_height = vae_width / vae_ratio
        vae_width = round(vae_width / 32) * 32
        vae_height = round(vae_height / 32) * 32

        return config, int(vl_width), int(vl_height), int(vae_width), int(vae_height)
