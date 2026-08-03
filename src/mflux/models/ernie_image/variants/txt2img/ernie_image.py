from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.ernie_image.ernie_image_initializer import ErnieImageInitializer
from mflux.models.ernie_image.latent_creator import ErnieLatentCreator
from mflux.models.ernie_image.model.ernie_text_encoder.prompt_encoder import ErniePromptEncoder
from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import ErnieMistralTextEncoder
from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer
from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.gpu_stream import with_gpu_stream
from mflux.utils.image_util import ImageUtil


class ErnieImage(nn.Module):
    vae: Flux2VAE
    text_encoder: ErnieMistralTextEncoder
    transformer: ErnieTransformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.ernie_image_turbo(),
    ):
        super().__init__()
        ErnieImageInitializer.init(
            model=self,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @with_gpu_stream
    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        if scheduler is None:
            scheduler = "linear"

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

        latents = self._prepare_latents(seed=seed, config=config)
        text_bth, text_lens = self._encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=config.guidance,
        )
        mx.eval(latents, text_bth, text_lens)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer, text_bth, text_lens, latents)

        for t in config.time_steps:
            try:
                sigma_t = config.scheduler.sigmas[t].reshape((1,))
                noise = predict(
                    latents=latents,
                    sigma=sigma_t,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    guidance=config.guidance,
                )
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        decoded = self._decode_latents(latents=latents)
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

    def _prepare_latents(self, *, seed: int, config: Config) -> mx.array:
        if config.image_path is None or config.image_strength is None or config.image_strength <= 0.0:
            return ErnieLatentCreator.create_noise(seed, config.height, config.width)

        pure_noise = ErnieLatentCreator.create_noise(seed, config.height, config.width)
        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        clean_latents = ErnieLatentCreator.pack_latents(encoded, height=config.height, width=config.width)
        clean_latents = ErnieLatentCreator.bn_normalize_latents(clean_latents, vae=self.vae)
        sigma = config.scheduler.sigmas[config.init_time_step]
        return LatentCreator.add_noise_by_interpolation(clean=clean_latents, noise=pure_noise, sigma=sigma)

    def _encode_prompts(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array]:
        cache_key = (prompt, negative_prompt, guidance)
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]

        tokenizer = self.tokenizers["ernie"]
        if guidance <= 1.0:
            prompts = [prompt]
        else:
            neg = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            prompts = [neg, prompt]

        text_bth, text_lens = ErniePromptEncoder.build_text_batch(
            prompts=prompts,
            tokenizer=tokenizer,
            text_encoder=self.text_encoder,
        )
        mx.eval(text_bth, text_lens)
        self.prompt_cache[cache_key] = (text_bth, text_lens)
        return text_bth, text_lens

    def _decode_latents(self, *, latents: mx.array) -> mx.array:
        return self.vae.decode_packed_latents(latents, tiling_config=self.tiling_config)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ErnieWeightDefinition,
        )

    @staticmethod
    def _predict(
        transformer: ErnieTransformer,
        text_bth: mx.array,
        text_lens: mx.array,
        latents: mx.array,
    ):
        H_lat, W_lat = latents.shape[-2:]
        cos, sin, attn_mask = transformer.get_pos_encoding(
            B=text_bth.shape[0],
            H=H_lat,
            W=W_lat,
            T=text_bth.shape[1],
            text_lens=text_lens,
        )
        mx.eval(cos, sin, attn_mask)

        def predict(
            latents: mx.array,
            sigma: mx.array,
            text_bth: mx.array,
            text_lens: mx.array,
            guidance: float,
        ) -> mx.array:
            timestep = sigma * 1000
            if text_bth.shape[0] == 1:
                return transformer(
                    hidden_states=latents,
                    timestep=mx.broadcast_to(timestep, (1,)),
                    text_bth=text_bth,
                    text_lens=text_lens,
                    cos=cos,
                    sin=sin,
                    attn_mask=attn_mask,
                )

            latent_input = mx.concatenate([latents, latents], axis=0)
            pred = transformer(
                hidden_states=latent_input,
                timestep=mx.broadcast_to(timestep, (2,)),
                text_bth=text_bth,
                text_lens=text_lens,
                cos=cos,
                sin=sin,
                attn_mask=attn_mask,
            )
            pred_uncond, pred_cond = pred[:1], pred[1:]
            return pred_uncond + guidance * (pred_cond - pred_uncond)

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)
