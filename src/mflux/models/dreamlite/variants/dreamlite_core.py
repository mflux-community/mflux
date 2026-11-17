from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.cli.defaults.defaults import model_inference_steps
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.dreamlite.dreamlite_initializer import DreamLiteInitializer
from mflux.models.dreamlite.latent_creator.dreamlite_latent_creator import DreamLiteLatentCreator
from mflux.models.dreamlite.model.dreamlite_text_encoder.dreamlite_text_encoder import DreamLiteTextEncoder
from mflux.models.dreamlite.model.dreamlite_text_encoder.prompt_encoder import DreamLitePromptEncoder
from mflux.models.dreamlite.model.dreamlite_unet.unet import DreamLiteUNet
from mflux.models.dreamlite.model.dreamlite_vae.vae import DreamLiteVAE
from mflux.models.dreamlite.weights.dreamlite_weight_definition import DreamLiteWeightDefinition
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class _DreamLiteCore(nn.Module):
    # Base runs 28 steps with guidance 3.5; Mobile is step-distilled to 4 steps without guidance.
    # Step counts come from MODEL_INFERENCE_STEPS
    DEFAULT_GUIDANCE = 3.5
    DEFAULT_IMAGE_GUIDANCE = 1.0
    NEGATIVE_KEY = "negative"
    # The side of the source image's copy that Qwen3-VL sees in an edit
    BASE_VISION_SIZE = 512
    MOBILE_VISION_SIZE = 256

    vae: DreamLiteVAE
    unet: DreamLiteUNet
    text_encoder: DreamLiteTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        compute_precision: mx.Dtype | None = None,
    ):
        super().__init__()
        DreamLiteInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config or ModelConfig.dreamlite(),
            compute_precision=compute_precision,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=DreamLiteWeightDefinition,
        )
        # Editing needs Qwen3-VL's image processor config next to the tokenizer
        self.processor.image_processor.save_pretrained(str(Path(base_path) / "processor"))

    def _generate(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None,
        height: int,
        width: int,
        guidance: float | None,
        image_guidance: float | None = None,
        negative_prompt: str | None = None,
        image: Image.Image | None = None,
        image_path: Path | str | None = None,
    ) -> GeneratedImage:
        # Shared by the txt2img and edit variants: image is the source image of an edit (None for txt2img)
        DreamLiteLatentCreator.check_size(height, width)
        num_inference_steps, guidance, image_guidance = self._resolve_sampling(num_inference_steps, guidance, image_guidance)  # fmt: off
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            scheduler="flow_match_euler_discrete",
        )
        config.scheduler.set_mu(FlowMatchEulerDiscreteScheduler._compute_linear_mu(config.image_seq_len))
        use_cfg = config.guidance > 1.0

        # 1. Encode the prompt, and the negative prompt when guidance is on. An edit's encodings hold the source image
        prompt_embeds, negative_embeds = self._encode_prompts(prompt, negative_prompt, image, use_cfg)

        # 2. Starting noise, kept in float32 between steps, and the latents of the context image: the source image of
        # an edit, zeros when generating
        latents = DreamLiteLatentCreator.create_noise(seed, config.height, config.width)
        empty_context = mx.zeros_like(latents)
        context = empty_context
        if image is not None:
            context = DreamLiteLatentCreator.encode_image(self.vae, image, config.height, config.width)

        # 3. Denoising loop
        predict = _DreamLiteCore._predict(
            unet=self.unet,
            prompt_embeds=prompt_embeds,
            negative_embeds=negative_embeds,
            context=context,
            empty_context=empty_context,
            size=(config.width, config.height),
            guidance=config.guidance,
            image_guidance=image_guidance if image is not None else None,
        )
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                velocity = predict(latents, config.scheduler.timesteps[t])
                latents = config.scheduler.step(noise=velocity, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)

        # 4. Decode. TAESDXL takes whole images: --vae-tiling sets a tiling config on the model, and it is not used
        decoded = VAEUtil.decode(self.vae, latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
            image_paths=[image_path] if image_path is not None else None,
            generation_parameters={
                **({"image_guidance": image_guidance} if image is not None and use_cfg else {}),
                **self.compute_precision.generation_parameters(),
            },
        )

    def _resolve_sampling(
        self,
        num_inference_steps: int | None,
        guidance: float | None,
        image_guidance: float | None,
    ) -> tuple[int, float, float]:
        if num_inference_steps is None:
            num_inference_steps = model_inference_steps(self.model_config.model_name, self.model_config.base_model)
        if not self.model_config.supports_guidance:
            # Mobile is distilled to run without guidance: like the reference pipeline, it takes no guidance value
            guidance = 1.0
        elif guidance is None:
            guidance = _DreamLiteCore.DEFAULT_GUIDANCE
        if image_guidance is None:
            image_guidance = _DreamLiteCore.DEFAULT_IMAGE_GUIDANCE
        return num_inference_steps, guidance, image_guidance

    def _encode_prompts(
        self,
        prompt: str,
        negative_prompt: str | None,
        image: Image.Image | None,
        use_cfg: bool,
    ) -> tuple[mx.array, mx.array | None]:
        negative = negative_prompt or ""
        if image is not None:
            # An edit's encodings depend on the source image, so they are not cached
            vision_size = self._vision_size()
            prompt_embeds = self._encode_edit(DreamLitePromptEncoder.edit_prompt(prompt), image, vision_size)
            negative_embeds = self._encode_edit(negative, image, vision_size) if use_cfg else None
            return prompt_embeds, negative_embeds
        # Cached under the prompt as given, which is the key MemorySaver looks for before it frees the text encoder
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = self._encode_generate(DreamLitePromptEncoder.generate_prompt(prompt))
        negative_embeds = None
        if use_cfg:
            negative_key = (_DreamLiteCore.NEGATIVE_KEY, negative)
            if negative_key not in self.prompt_cache:
                self.prompt_cache[negative_key] = self._encode_generate(negative)
            negative_embeds = self.prompt_cache[negative_key]
        return self.prompt_cache[prompt], negative_embeds

    def _encode_generate(self, text: str) -> mx.array:
        embeds = DreamLitePromptEncoder.encode_generate(text, self.processor, self.text_encoder)
        mx.eval(embeds)
        return embeds

    def _encode_edit(self, text: str, image: Image.Image, vision_size: int) -> mx.array:
        embeds = DreamLitePromptEncoder.encode_edit(text, image, vision_size, self.processor, self.text_encoder)
        mx.eval(embeds)
        return embeds

    def _vision_size(self) -> int:
        # Mobile, the model without guidance, was distilled with the smaller copy
        if self.model_config.supports_guidance:
            return _DreamLiteCore.BASE_VISION_SIZE
        return _DreamLiteCore.MOBILE_VISION_SIZE

    @staticmethod
    def _predict(
        unet: DreamLiteUNet,
        prompt_embeds: mx.array,
        negative_embeds: mx.array | None,
        context: mx.array,
        empty_context: mx.array,
        size: tuple[int, int],
        guidance: float,
        image_guidance: float | None,
    ):
        def velocity(latents: mx.array, timestep: mx.array, embeds: mx.array, context_latents: mx.array) -> mx.array:
            # The UNet sees the latents and the context image side by side; only the latents' half is a prediction
            model_input = DreamLiteLatentCreator.with_context(latents, context_latents)
            prediction = unet(model_input, timestep, embeds, size)
            return prediction[..., : latents.shape[-1]].astype(mx.float32)

        # One pass for each guidance term, in sequence: compiled, they run as fast as one batch of them
        def predict(latents: mx.array, timestep: mx.array) -> mx.array:
            prompted = velocity(latents, timestep, prompt_embeds, context)
            if negative_embeds is None:
                return prompted
            # The negative prompt's pass on the same context: text guidance alone when generating
            negative = velocity(latents, timestep, negative_embeds, context)
            guided = negative + guidance * (prompted - negative)
            if image_guidance is None or image_guidance == 1.0:
                # At image guidance 1.0 an edit's third pass, without the source image, cancels out
                return guided
            unconditional = velocity(latents, timestep, negative_embeds, empty_context)
            return guided + (image_guidance - 1.0) * (negative - unconditional)

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)
