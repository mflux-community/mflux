from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class QwenImage21(nn.Module):
    vae: Qwen21VAE
    transformer: Qwen21Transformer
    text_encoder: Qwen21TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.qwen_image_21(),
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ):
        super().__init__()
        Qwen21Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    @staticmethod
    def _teacache_skip_steps(transformer: Qwen21Transformer, config: Config, ratio: float) -> frozenset[int]:
        """TeaCache-style selection of denoise steps whose transformer call is skipped.

        The timestep-embedding signal depends only on the sigma schedule, so it is
        computed for every step up front and the ``ratio``-fraction of steps with the
        smallest step-to-step change — inside a protected window that keeps the first
        and last 10% of the run unskipped — reuse the previous noise instead of
        running the transformer. (Liu et al., "Timestep Embedding Tells: It's All
        You Need for Accelerating DiT-based Diffusion Models", ICLR 2025.)
        """
        if not 0 < ratio < 1:
            raise ValueError(f"teacache_ratio must be within (0, 1), got {ratio}")
        steps = list(config.time_steps)
        total = len(steps)
        if total < 10:
            return frozenset()
        sigmas = config.scheduler.sigmas
        timesteps = mx.array([QwenImage21._step_timestep(sigmas, t) for t in steps], dtype=mx.float32)
        signals = transformer.time_text_embed(timesteps)
        mx.eval(signals)
        signals = signals.astype(mx.float32)
        first = steps[0]
        low = first + total // 10  # keep the first and last 10% of the run unskipped
        high = first + total - total // 10
        eligible = [i for i, t in enumerate(steps) if low <= t < high]
        if not eligible:
            return frozenset()
        diffs = [mx.sqrt(mx.sum(mx.square(signals[i] - signals[i - 1]))).item() for i in eligible]
        count = min(round(ratio * total), len(eligible))
        chosen = sorted(range(len(eligible)), key=lambda j: diffs[j])[:count]
        return frozenset(steps[eligible[j]] for j in chosen)

    @staticmethod
    def _step_timestep(sigmas: mx.array, t: int) -> float:
        """Mirror Qwen21Transformer._compute_timestep's int-step sigma lookup."""
        if t < len(sigmas):
            return float(sigmas[t])
        return t / 1000.0 if t > 1.0 else float(t)

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 40,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
        negative_prompt: str | None = None,
        teacache_ratio: float | None = None,
    ) -> GeneratedImage:
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

        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=Qwen21LatentCreator,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                image_path=config.image_path,
                tiling_config=self.tiling_config,
            ),
        )
        # img2img noise interpolation promotes to fp32; keep the stream in model precision
        latents = latents.astype(ModelConfig.precision)

        prompt_embeds, prompt_mask = Qwen21PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            tokenizer=self.tokenizers["qwen21"],
            text_encoder=self.text_encoder,
        )
        # like the reference pipeline, a missing (or empty) negative prompt disables true CFG
        do_true_cfg = config.guidance > 1.0 and bool(negative_prompt)
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_mask = Qwen21PromptEncoder.encode_prompt(
                prompt=negative_prompt,
                prompt_cache=self.prompt_cache,
                tokenizer=self.tokenizers["qwen21"],
                text_encoder=self.text_encoder,
            )

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        skip_steps = (
            QwenImage21._teacache_skip_steps(self.transformer, config, teacache_ratio)
            if teacache_ratio is not None
            else frozenset()
        )
        previous_noise: mx.array | None = None

        try:
            for t in config.time_steps:
                try:
                    latents = config.scheduler.scale_model_input(latents, t)
                    if t in skip_steps and previous_noise is not None:
                        # TeaCache-style step reuse: the timestep-embedding signal for this
                        # step is close to the previous one, so reuse its noise prediction
                        # and skip the transformer (and any true-CFG pass) entirely.
                        noise = previous_noise
                    else:
                        noise = self.transformer(
                            t=t,
                            config=config,
                            hidden_states=latents,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_mask=prompt_mask,
                        )
                        if do_true_cfg:
                            noise_negative = self.transformer(
                                t=t,
                                config=config,
                                hidden_states=latents,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_hidden_states_mask=negative_prompt_mask,
                            )
                            noise = noise_negative + config.guidance * (noise - noise_negative)
                        previous_noise = noise

                    latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                    ctx.in_loop(t, latents)
                    mx.eval(latents)

                except KeyboardInterrupt:  # noqa: PERF203
                    ctx.interruption(t, latents)
                    raise StopImageGenerationException(
                        f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                    )
        finally:
            # the text-prefix K/V cache is O(100 MB) per prompt: free it when the loop ends
            self.transformer.clear_text_cache()

        ctx.after_loop(latents)

        latents = Qwen21LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Qwen21WeightDefinition,
        )
