import shutil
from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.cli.defaults.defaults import model_inference_steps
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.fibo2.fibo2_initializer import Fibo2Initializer
from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_images import Fibo2TextImages
from mflux.models.fibo2.model.fibo2_text_encoder.prompt_encoder import Fibo2PromptEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class _Fibo2Core(nn.Module):
    # Dev runs with guidance 5; Edit Turbo is distilled to run without. Step counts come from MODEL_INFERENCE_STEPS
    DEFAULT_GUIDANCE = 5.0
    # A generation shows Qwen3-VL the images of the caption's text only once the noise level is at or below this
    TEXT_IMAGES_MAX_NOISE_LEVEL = 0.9

    vae: Flux2VAE
    transformer: Fibo2Transformer
    text_encoder: Fibo2TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ):
        super().__init__()
        Fibo2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config or ModelConfig.fibo2(),
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Fibo2WeightDefinition,
        )
        # Editing needs Qwen3-VL's image processor config next to the tokenizer, and text images need the fonts
        self.processor.image_processor.save_pretrained(str(Path(base_path) / "processor"))
        if self.fonts_dir is not None:
            shutil.copytree(self.fonts_dir, Path(base_path) / "fonts", dirs_exist_ok=True)

    def _generate(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None,
        height: int,
        width: int,
        guidance: float | None,
        negative_prompt: str | None = None,
        images: list[Image.Image] | None = None,
        text_encoder_images: list[Image.Image] | None = None,
        image_paths: list[Path | str] | None = None,
        mask_path: Path | str | None = None,
        render_text_images: bool = True,
    ) -> GeneratedImage:
        # Shared by the txt2img and edit variants: images are the crops to edit (None for txt2img)
        num_inference_steps, guidance = self._resolve_sampling(num_inference_steps, guidance)
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            scheduler="flow_match_euler_discrete",
        )
        config.scheduler.set_mu(FlowMatchEulerDiscreteScheduler._compute_linear_mu(config.image_seq_len))

        # 1. Encode the prompt (an edit's with its images), and the negative prompt, without them, when guidance is on
        prompt, prompt_embeds, base_embeds = self._encode_prompts(
            prompt, images, text_encoder_images, render_text_images
        )
        negative_embeds = None
        if config.guidance > 1.0:
            negative_embeds = self._encode_prompt(negative_prompt or Fibo2PromptEncoder.NULL_CAPTION)

        # 2. Starting noise, kept in float32 between steps, and the latents of the images to edit
        latents = Fibo2LatentCreator.create_noise(seed, config.height, config.width)
        context_latents = [Fibo2LatentCreator.encode_image(self.vae, image, self.tiling_config) for image in images or []]  # fmt: off

        # 3. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                sigma = config.scheduler.sigmas[t : t + 1]
                embeds = _Fibo2Core._embeds_for_step(float(sigma[0]), prompt_embeds, base_embeds)
                velocity = self._predict(latents, sigma, embeds, negative_embeds, config.guidance, context_latents)
                latents = config.scheduler.step(noise=velocity, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)

        # 4. Decode
        decoded = Fibo2LatentCreator.decode(self.vae, latents, tiling_config=self.tiling_config)
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
            image_paths=image_paths,
            masked_image_path=mask_path,
        )

    @staticmethod
    def _embeds_for_step(sigma: float, prompt_embeds: mx.array, base_embeds: mx.array | None) -> mx.array:
        # A generation's first steps, above the noise level, run on the caption without its text images
        if base_embeds is None or sigma <= _Fibo2Core.TEXT_IMAGES_MAX_NOISE_LEVEL:
            return prompt_embeds
        return base_embeds

    def _resolve_sampling(self, num_inference_steps: int | None, guidance: float | None) -> tuple[int, float]:
        if num_inference_steps is None:
            num_inference_steps = model_inference_steps(self.model_config.model_name, self.model_config.base_model)
        if guidance is None:
            guidance = _Fibo2Core.DEFAULT_GUIDANCE if self.model_config.supports_guidance else 1.0
        return num_inference_steps, guidance

    def _encode_prompts(
        self,
        prompt: str,
        images: list[Image.Image] | None,
        text_encoder_images: list[Image.Image] | None,
        render_text_images: bool,
    ) -> tuple[str, mx.array, mx.array | None]:
        # A generation with text images also gets the embeddings its first steps run on; an edit uses them throughout
        if images:
            caption, _ = Fibo2PromptEncoder.edit_caption(prompt)
            text_images = self._text_images(caption) if render_text_images else []
            prompt = Fibo2PromptEncoder.edit_prompt(prompt, len(images), text_images=bool(text_images))
            return prompt, self._encode_prompt(prompt, [*text_encoder_images, *text_images]), None
        prompt = Fibo2PromptEncoder.format_prompt(prompt)
        text_images = self._text_images(Fibo2PromptEncoder.caption(prompt)) if render_text_images else []
        if not text_images:
            return prompt, self._encode_prompt(prompt), None
        marked = Fibo2PromptEncoder.mark_text_images(prompt, Fibo2PromptEncoder.VISION_MARKER)
        base = Fibo2PromptEncoder.mark_text_images(prompt, "")
        return prompt, self._encode_prompt(marked, text_images), self._encode_prompt(base)

    def _text_images(self, caption: dict | str | None) -> list[Image.Image]:
        if not isinstance(caption, dict):
            return []
        return Fibo2TextImages.draw_for_caption(caption, self.fonts_dir)

    def _encode_prompt(self, prompt: str, images: list[Image.Image] | None = None) -> mx.array:
        # A prompt with images is encoded each time; text alone is cached
        if images:
            return Fibo2PromptEncoder.encode_prompt(prompt, self.processor, self.text_encoder, images)
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = Fibo2PromptEncoder.encode_prompt(prompt, self.processor, self.text_encoder)
        return self.prompt_cache[prompt]

    def _predict(
        self,
        latents: mx.array,
        sigma: mx.array,
        prompt_embeds: mx.array,
        negative_embeds: mx.array | None,
        guidance: float,
        context_latents: list[mx.array],
    ) -> mx.array:
        velocity = self.transformer(latents, sigma, prompt_embeds, context_latents=context_latents).astype(mx.float32)
        if negative_embeds is None:
            return velocity
        # The negative prompt runs as a second pass, as in the reference; the images to edit condition both passes
        unconditional = self.transformer(latents, sigma, negative_embeds, context_latents=context_latents)
        unconditional = unconditional.astype(mx.float32)
        return unconditional + guidance * (velocity - unconditional)
