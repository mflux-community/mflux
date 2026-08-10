from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Krea2(nn.Module):
    vae: QwenVAE
    transformer: Krea2Transformer
    text_encoder: Krea2TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        Krea2Initializer.init(
            model=self,
            model_config=model_config or ModelConfig.krea2(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
    ) -> GeneratedImage:
        resolved_scheduler = Krea2._resolve_scheduler(scheduler)

        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=resolved_scheduler,
        )

        sigmas = config.scheduler.sigmas
        latents = self._prepare_latents(seed=seed, config=config)
        embeds, neg_embeds = self._encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
        )
        mx.eval(latents, embeds)
        if neg_embeds is not None:
            mx.eval(neg_embeds)

        stepper = Krea2Sampler.make_stepper(resolved_scheduler, sigmas, seed)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer, embeds, neg_embeds, guidance)

        for t in config.time_steps:
            try:
                ts = sigmas[t].reshape(1)
                v = predict(latents=latents, timestep=ts)
                denoised = latents - sigmas[t] * v
                latents = stepper.step(t, latents, v, denoised)
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
            generation_time=config.time_steps.format_dict["elapsed"],
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            negative_prompt=negative_prompt,
            image_path=config.image_path,
            image_strength=config.image_strength,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Krea2WeightDefinition,
        )

    def _encode_prompts(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array | None]:
        return Krea2PromptEncoder.encode_prompt_pair(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
            tokenizer=self.tokenizers["qwen3vl"],
            text_encoder=self.text_encoder,
            prompt_cache=self.prompt_cache,
        )

    def _prepare_latents(self, *, seed: int, config: Config) -> mx.array:
        if config.image_path is None or config.image_strength is None or config.image_strength <= 0.0:
            return Krea2LatentCreator.create_noise(seed, config.height, config.width)

        pure_noise = Krea2LatentCreator.create_noise(seed, config.height, config.width)
        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        clean_latents = Krea2LatentCreator.pack_latents(encoded, config.height, config.width)
        sigma = float(config.scheduler.sigmas[config.init_time_step])
        return LatentCreator.add_noise_by_interpolation(clean=clean_latents, noise=pure_noise, sigma=sigma)

    def _decode_latents(self, *, latents: mx.array) -> mx.array:
        return self.vae.decode(latents)

    @staticmethod
    def _predict(
        transformer: Krea2Transformer,
        embeds: mx.array,
        neg_embeds: mx.array | None,
        guidance: float,
    ):
        def predict(latents: mx.array, timestep: mx.array) -> mx.array:
            v = transformer(latents, timestep, embeds)
            if neg_embeds is not None:
                v_neg = transformer(latents, timestep, neg_embeds)
                v = v_neg + guidance * (v - v_neg)
            return v

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)

    @staticmethod
    def _resolve_scheduler(scheduler: str | None) -> str:
        if scheduler is None or scheduler == "linear":
            return "er_sde"
        if scheduler in ("er_sde", "euler"):
            return scheduler
        raise ValueError(f"Unknown Krea-2 scheduler {scheduler!r}. Expected 'er_sde' or 'euler'.")
