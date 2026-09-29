import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from PIL import Image

from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.prompt_encoder import QwenImage21PromptEncoder
from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
from mflux.models.qwen21.reference.qwen_image21_initializer import QwenImage21Initializer
from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.exif_orientation import open_oriented
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class QwenImage21Edit(nn.Module):
    def __init__(
        self, quantize: int | None = None, model_path: str | None = None, model_config: ModelConfig | None = None
    ):
        super().__init__()
        QwenImage21Initializer.init(self, model_config or ModelConfig.qwen_image_21(), quantize, model_path)

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = MODEL_INFERENCE_STEPS["qwen-image-2.1"],
        height: int | None = None,
        width: int | None = None,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        image_paths: list[str | Path] | None = None,
        output_resolution: int = 1024,
        use_kv_cache: bool = True,
    ) -> GeneratedImage:
        image_paths = image_paths or []
        if len(image_paths) > 10:
            raise ValueError("Qwen-Image-2.1 supports at most 10 reference images.")
        QwenImage21LatentCreator.validate_resolution(output_resolution)
        images = [open_oriented(path).convert("RGBA") for path in image_paths]
        ratio = images[-1].width / images[-1].height if images else 1.0
        default_width, default_height = QwenImage21LatentCreator.dimensions(output_resolution, ratio)
        width = default_width if width is None else width
        height = default_height if height is None else height
        QwenImage21LatentCreator.validate(width, height, num_inference_steps, len(images))
        if guidance < 1 or not np.isfinite(guidance):
            raise ValueError("guidance must be finite and at least 1.")
        if guidance > 1 and negative_prompt is None:
            raise ValueError("Provide negative_prompt (which may be empty) to enable guidance greater than 1.")
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            guidance=guidance,
        )
        images = [
            image.resize(
                QwenImage21LatentCreator.dimensions(output_resolution, image.width / image.height),
                Image.Resampling.LANCZOS,
            )
            for image in images
        ]
        size = self.processor.image_processor.size
        for image in images:
            if not size["shortest_edge"] <= image.width * image.height <= size["longest_edge"]:
                raise ValueError(
                    "Reference image area falls outside the vision processor's range; "
                    "adjust output_resolution (the default is 1024)."
                )
        prompt_embeds, slots = self._encode_prompt(prompt, images)
        negative = self._encode_prompt(negative_prompt, images) if guidance > 1 else None
        mx.eval(prompt_embeds)
        shapes = [(1, img.height // 16, img.width // 16) for img in images] + [(1, height // 16, width // 16)]
        layout = QwenImage21Layout.create(slots, shapes, self.transformer.axes)
        negative_layout = QwenImage21Layout.create(negative[1], shapes, self.transformer.axes) if negative else None
        conditions = []
        for image in images:
            pixels = mx.array(np.asarray(image).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
            conditions.append(
                QwenImage21LatentCreator.pack_latents(self.vae.encode(pixels)).astype(prompt_embeds.dtype)
            )
        condition_latents = mx.concatenate(conditions, axis=1) if conditions else None
        mx.random.seed(seed)
        latents = mx.random.normal((1, 64, 1, height // 16, width // 16)).astype(prompt_embeds.dtype)
        latents = QwenImage21LatentCreator.pack_latents(latents)
        cache = [] if use_kv_cache else None
        negative_cache = [] if use_kv_cache else None
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                model_input = mx.concatenate([condition_latents, latents], axis=1) if conditions else latents
                # Match diffusers: cast the 0..1000 timestep before dividing by 1000.
                timestep = (config.scheduler.sigmas[t : t + 1] * 1000).astype(latents.dtype) / 1000
                noise = self.transformer(model_input, prompt_embeds, timestep, layout, cache)
                if negative is not None:
                    uncond = self.transformer(model_input, negative[0], timestep, negative_layout, negative_cache)
                    noise = uncond + guidance * (noise - uncond)
                # The upstream Euler update accumulates in fp32 and casts back afterward.
                sigma = config.scheduler.sigmas
                latents = (latents.astype(mx.float32) + (sigma[t + 1] - sigma[t]) * noise.astype(mx.float32)).astype(
                    latents.dtype
                )
                mx.eval(latents)
                ctx.in_loop(t, latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{num_inference_steps}"
                ) from None
        ctx.after_loop(latents)
        del cache, negative_cache
        unpacked = QwenImage21LatentCreator.unpack_latents(latents, height, width).astype(mx.float32)
        decoded = VAEUtil.decode(self.vae, unpacked, self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            image_paths=image_paths,
            negative_prompt=negative_prompt,
            generation_parameters={"use_kv_cache": use_kv_cache, "output_resolution": output_resolution},
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path, QwenImage21WeightDefinition)
        destination = Path(base_path)
        for name, config in self._component_configs.items():
            (destination / name / "config.json").write_text(json.dumps(config, indent=2))
        self.processor.save_pretrained(str(destination / "processor"))
        for relative in ("model_index.json", "scheduler/scheduler_config.json"):
            source = Path(self._checkpoint_path) / relative
            if source.exists() and source.resolve() != (destination / relative).resolve():
                (destination / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination / relative)

    def _encode_prompt(self, prompt: str, images: list[Image.Image]) -> tuple[mx.array, mx.array]:
        if not images and prompt in self.prompt_cache:
            return self.prompt_cache[prompt]
        if self.text_encoder is None:
            raise RuntimeError("The text encoder was released by the memory saver; reload the model for a new prompt.")
        encoded = QwenImage21PromptEncoder.encode(prompt, images, self.processor, self.text_encoder)
        if not images:
            self.prompt_cache[prompt] = encoded
        return encoded
