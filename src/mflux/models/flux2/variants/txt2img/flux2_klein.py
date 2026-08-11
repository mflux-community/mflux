from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux2Klein(nn.Module):
    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        Flux2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            model_config=model_config or ModelConfig.flux2_klein_4b(),
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
    ) -> GeneratedImage:
        # 0. Create a new config based on the model type and input parameters
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )
        # 1. Encode prompt(s)
        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=" ",
            guidance=guidance,
        )

        # 2. Prepare latents (txt2img or img2img)
        latents, latent_ids, latent_height, latent_width = self._prepare_generation_latents(
            seed=seed,
            config=config,
        )

        # 3. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)
        for t in config.time_steps:
            try:
                # 3.t Predict the noise
                noise = predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance=guidance,
                    timestep=config.scheduler.timesteps[t],
                )

                # 4.t Take one denoise step
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
                )

                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # 5. Decode latents
        packed_latents = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)  # fmt: off
        decoded = self.vae.decode_packed_latents(packed_latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=None,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        prompt_embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            negative_prompt_embeds, negative_text_ids = Flux2PromptEncoder.encode_prompt(
                prompt=negative_prompt,
                tokenizer=self.tokenizers["qwen3"],
                text_encoder=self.text_encoder,
                num_images_per_prompt=1,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27),
            )
        return prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids

    def _prepare_generation_latents(
        self,
        *,
        seed: int,
        config: Config,
    ) -> tuple[mx.array, mx.array, int, int]:
        if config.image_path is None or config.image_strength is None or config.image_strength <= 0.0:
            return Flux2LatentCreator.prepare_packed_latents(
                seed=seed,
                height=config.height,
                width=config.width,
                batch_size=1,
            )
        return self._prepare_img2img_latents(seed=seed, config=config)

    def _prepare_img2img_latents(
        self,
        *,
        seed: int,
        config: Config,
    ) -> tuple[mx.array, mx.array, int, int]:
        noise_latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )

        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
        encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
        encoded = self._match_latent_spatial_size(
            encoded=encoded,
            target_height=latent_height * 2,
            target_width=latent_width * 2,
        )
        encoded = Flux2LatentCreator.patchify_latents(encoded)
        encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=self.vae)
        clean_latents = Flux2LatentCreator.pack_latents(encoded)

        sigma = config.scheduler.sigmas[config.init_time_step]
        latents = LatentCreator.add_noise_by_interpolation(clean=clean_latents, noise=noise_latents, sigma=sigma)
        return latents, latent_ids, latent_height, latent_width

    @staticmethod
    def _match_latent_spatial_size(
        *,
        encoded: mx.array,
        target_height: int,
        target_width: int,
    ) -> mx.array:
        _, _, height, width = encoded.shape
        if height != target_height:
            if height > target_height:
                offset = (height - target_height) // 2
                encoded = encoded[:, :, offset : offset + target_height, :]
            else:
                pad_total = target_height - height
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                encoded = mx.pad(encoded, ((0, 0), (0, 0), (pad_before, pad_after), (0, 0)))
        if width != target_width:
            if width > target_width:
                offset = (width - target_width) // 2
                encoded = encoded[:, :, :, offset : offset + target_width]
            else:
                pad_total = target_width - width
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                encoded = mx.pad(encoded, ((0, 0), (0, 0), (0, 0), (pad_before, pad_after)))
        return encoded

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Flux2KleinWeightDefinition,
        )

    @staticmethod
    def _predict(transformer):
        def predict(
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
        ):
            noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,
            )
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                )
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)
