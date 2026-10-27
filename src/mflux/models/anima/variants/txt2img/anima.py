import copy

import mlx.core as mx
from mlx import nn

from mflux.models.anima.anima_initializer import AnimaInitializer
from mflux.models.anima.latent_creator.anima_latent_creator import AnimaLatentCreator
from mflux.models.anima.model.anima_scheduler import AnimaSampler
from mflux.models.anima.model.anima_text_encoder.anima_prompt_encoder import AnimaPromptEncoder
from mflux.models.anima.model.anima_transformer.anima_transformer import AnimaTransformer
from mflux.models.anima.weights.anima_weight_definition import AnimaWeightDefinition
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Anima(nn.Module):
    vae: QwenVAE
    transformer: AnimaTransformer
    text_encoder: Qwen3TextEncoder

    # Steps and CFG from the model card: Turbo is distilled for CFG 1 and 8 to 12 steps, Aesthetic and Base
    # take 30 to 50 steps at CFG 4 to 5.
    DEFAULT_STEPS = {"anima-turbo": 10, "anima-aesthetic": 30, "anima-base": 30}
    DEFAULT_GUIDANCE = {"anima-turbo": 1.0, "anima-aesthetic": 4.0, "anima-base": 4.0}

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        compute_precision: mx.Dtype | None = None,
    ):
        super().__init__()
        AnimaInitializer.init(
            model=self,
            model_config=model_config or ModelConfig.anima_turbo(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            compute_precision=compute_precision,
        )

    @property
    def variant(self) -> str:
        return self.model_config.aliases[0]

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        scheduler: str | None = None,
    ) -> GeneratedImage:
        AnimaLatentCreator.check_size(height=height, width=width)
        sampler_name = AnimaSampler.resolve(scheduler)
        config = Config(
            model_config=self._metadata_model_config(),
            num_inference_steps=num_inference_steps or Anima.DEFAULT_STEPS[self.variant],
            height=height,
            width=width,
            guidance=self.guidance_or_default(guidance),
            scheduler="anima",
        )
        sampler = AnimaSampler.make(sampler_name, config.scheduler.sigmas, seed)
        positive = self._encode_prompt(prompt)
        negative = self._encode_prompt(negative_prompt or "") if config.guidance != 1.0 else None
        latents = AnimaLatentCreator.create_noise(seed=seed, height=config.height, width=config.width)

        predict = Anima._predict(self.transformer, positive, negative, config.guidance)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                sigma = sampler.sigma(t)
                velocity = predict(latents, mx.reshape(sigma, (1,)))
                denoised = latents - sigma * velocity
                latents = sampler.step(t, latents, velocity, denoised)
                ctx.in_loop(t, latents, denoised=denoised)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        # The predict closure holds the transformer. Delete it so that --low-ram can free the transformer before decode.
        del predict
        ctx.after_loop(latents)

        decoded = VAEUtil.decode(vae=self.vae, latent=latents.astype(ModelConfig.precision), tiling_config=self.tiling_config)  # fmt: off
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            negative_prompt=negative_prompt,
            generation_time=config.time_steps.format_dict["elapsed"],
            generation_parameters=self.compute_precision.generation_parameters(),
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=AnimaWeightDefinition,
        )

    def guidance_or_default(self, guidance: float | None) -> float:
        return Anima.DEFAULT_GUIDANCE[self.variant] if guidance is None else float(guidance)

    @staticmethod
    def _predict(
        transformer: AnimaTransformer,
        positive: mx.array,
        negative: mx.array | None,
        guidance: float,
    ):
        # Two passes for CFG: compiled, they run faster than one batch of two.
        def predict(latents: mx.array, sigma: mx.array) -> mx.array:
            cond = transformer(latents, sigma, positive).astype(mx.float32)
            if negative is None:
                return cond
            uncond = transformer(latents, sigma, negative).astype(mx.float32)
            return uncond + guidance * (cond - uncond)

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)

    def _encode_prompt(self, prompt: str) -> mx.array:
        if prompt not in self.prompt_cache:
            context = AnimaPromptEncoder.encode(
                prompt=prompt,
                qwen_tokenizer=self.tokenizers["qwen3"],
                t5_tokenizer=self.tokenizers["t5"],
                text_encoder=self.text_encoder,
                transformer=self.transformer,
            )
            mx.eval(context)
            self.prompt_cache[prompt] = context
        return self.prompt_cache[prompt]

    def _metadata_model_config(self) -> ModelConfig:
        # The three variants share one repo id, which is all a metadata sidecar records as the model. Naming the
        # entry as base_model lets --config-from-metadata replay an Aesthetic or Base run as that variant instead
        # of as the default Turbo.
        if self.variant == "anima-turbo":
            return self.model_config
        model_config = copy.copy(self.model_config)
        model_config.base_model = self.variant
        return model_config
