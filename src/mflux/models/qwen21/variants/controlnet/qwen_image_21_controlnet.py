import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from PIL import Image

from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.prompt_encoder import QwenImage21PromptEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.text_encoder import QwenImage21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.model.qwen21_vae.vae import QwenImage21VAE
from mflux.models.qwen21.qwen_image21_initializer import QwenImage21Initializer
from mflux.models.qwen21.variants.controlnet.qwen_image21_controlnet_transformer import QwenImage21ControlNet
from mflux.models.qwen21.weights.qwen_image21_controlnet_weight_definition import (
    QwenImage21ControlnetWeightDefinition,
)
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.exif_orientation import open_oriented
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class QwenImage21Controlnet(nn.Module):
    # Qwen-Image-2.1 with alibaba-pai's Fun ControlNet-Union: text to image steered by a control image
    # (canny, depth, pose, ...), and inpainting through the same branch when a source and a mask are given.
    # It loads the same components as QwenImage21Edit and adds the control branch beside the transformer.
    vae: QwenImage21VAE
    transformer: QwenImage21Transformer
    text_encoder: QwenImage21TextEncoder
    controlnet: QwenImage21ControlNet

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        controlnet_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        compute_precision: mx.Dtype | None = None,
    ):
        super().__init__()
        QwenImage21Initializer.init_controlnet(
            self,
            model_config or ModelConfig.qwen_image_21_controlnet_union(),
            quantize,
            model_path,
            controlnet_path,
            lora_paths,
            lora_scales,
            bake_lora,
            compute_precision=compute_precision,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        controlnet_image_path: str | Path | Image.Image | None = None,
        controlnet_strength: float = 1.0,
        num_inference_steps: int = MODEL_INFERENCE_STEPS["qwen-image-2.1"],
        height: int | None = None,
        width: int | None = None,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        output_resolution: int = 1024,
        image_path: str | Path | Image.Image | None = None,
        mask_image: str | Path | Image.Image | None = None,
    ) -> GeneratedImage:
        if controlnet_image_path is None and image_path is None:
            raise ValueError("Give a control image, or a source image and a mask to inpaint.")
        if (image_path is None) != (mask_image is None):
            raise ValueError("Inpainting needs both image_path (the source) and mask_image.")
        control = QwenImage21Controlnet._open(controlnet_image_path)
        source = QwenImage21Controlnet._open(image_path)
        shape = control or source
        default_width, default_height = QwenImage21LatentCreator.dimensions(
            output_resolution, shape.width / shape.height
        )
        width = default_width if width is None else width
        height = default_height if height is None else height
        QwenImage21LatentCreator.validate(width, height, num_inference_steps, 0)
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
            controlnet_strength=controlnet_strength,
        )
        # Under control the joint stream carries only the prompt and the target: the source of an inpaint
        # goes in through the control input, never through the text encoder's vision slots.
        prompt_embeds, slots = self._encode_prompt(prompt)
        negative = self._encode_prompt(negative_prompt) if guidance > 1 else None
        mx.eval(prompt_embeds)
        shapes = [(1, height // 16, width // 16)]
        layout = QwenImage21Layout.create(slots, shapes, self.transformer.axes)
        negative_layout = QwenImage21Layout.create(negative[1], shapes, self.transformer.axes) if negative else None
        control_context = self._control_context(control, source, mask_image, width, height).astype(prompt_embeds.dtype)
        mx.eval(control_context)
        mx.random.seed(seed)
        latents = mx.random.normal((1, 64, 1, height // 16, width // 16)).astype(prompt_embeds.dtype)
        latents = QwenImage21LatentCreator.pack_latents(latents)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                # Match diffusers: cast the 0..1000 timestep before dividing by 1000.
                timestep = (config.scheduler.sigmas[t : t + 1] * 1000).astype(latents.dtype) / 1000
                noise = self.controlnet(
                    self.transformer, latents, prompt_embeds, timestep, layout, control_context, controlnet_strength
                )
                if negative is not None:
                    uncond = self.controlnet(
                        self.transformer,
                        latents,
                        negative[0],
                        timestep,
                        negative_layout,
                        control_context,
                        controlnet_strength,
                    )
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
        unpacked = QwenImage21LatentCreator.unpack_latents(latents, height, width).astype(mx.float32)
        decoded = VAEUtil.decode(self.vae, unpacked, self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            generation_time=config.time_steps.format_dict["elapsed"],
            controlnet_image_path=QwenImage21Controlnet._recorded(controlnet_image_path),
            image_path=QwenImage21Controlnet._recorded(image_path),
            masked_image_path=QwenImage21Controlnet._recorded(mask_image),
            negative_prompt=negative_prompt,
            generation_parameters=self.compute_precision.generation_parameters(),
        )

    def save_model(self, base_path: str) -> None:
        # The base checkpoint as QwenImage21Edit saves it, with the control branch under controlnet/.
        ModelSaver.save_model(self, self.bits, base_path, QwenImage21ControlnetWeightDefinition)
        destination = Path(base_path)
        for name, config in self._component_configs.items():
            (destination / name / "config.json").write_text(json.dumps(config, indent=2))
        self.processor.save_pretrained(str(destination / "processor"))
        for relative in ("model_index.json", "scheduler/scheduler_config.json"):
            source = Path(self._checkpoint_path) / relative
            if source.exists() and source.resolve() != (destination / relative).resolve():
                (destination / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination / relative)

    def _encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        # Text only: no image goes through the text encoder's vision slots here.
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt]
        if self.text_encoder is None:
            raise RuntimeError("The text encoder was released by the memory saver; reload the model for a new prompt.")
        self.prompt_cache[prompt] = QwenImage21PromptEncoder.encode(prompt, [], self.processor, self.text_encoder)
        return self.prompt_cache[prompt]

    def _control_context(
        self,
        control: Image.Image | None,
        source: Image.Image | None,
        mask_image: str | Path | Image.Image | None,
        width: int,
        height: int,
    ) -> mx.array:
        # 129 channels per latent token, as the reference pipeline packs them: the control image's latents,
        # the keep-mask (1 = keep, 0 = regenerate) and the latents of the source with the region to regenerate
        # blanked. Without a mask everything is regenerated; an absent image contributes zeros.
        latent_height, latent_width = height // 16, width // 16
        repaint = np.ones((height, width), dtype=np.float32)
        if mask_image is not None:
            mask = QwenImage21Controlnet._open(mask_image, mode="L").resize((width, height), Image.Resampling.LANCZOS)
            repaint = (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
        empty = mx.zeros((1, 64, 1, latent_height, latent_width), dtype=mx.float32)
        control_latents = empty if control is None else self._encode(control, width, height)
        source_latents = empty if source is None else self._encode(source, width, height, keep=1 - repaint)
        # nearest-neighbour to the latent grid, like torch's F.interpolate(mode="nearest")
        rows = (np.arange(latent_height) * height // latent_height).astype(int)
        columns = (np.arange(latent_width) * width // latent_width).astype(int)
        keep = mx.array((1 - repaint)[rows][:, columns]).reshape(1, 1, 1, latent_height, latent_width)
        context = mx.concatenate([control_latents.astype(mx.float32), keep, source_latents.astype(mx.float32)], axis=1)
        return QwenImage21LatentCreator.pack_latents(context)

    def _encode(self, image: Image.Image, width: int, height: int, keep: np.ndarray | None = None) -> mx.array:
        # RGB at the target size in [-1, 1], blanked where keep is 0, then an opaque alpha: the VAE reads RGBA.
        pixels = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)).astype(np.float32)
        pixels = pixels / 127.5 - 1
        if keep is not None:
            pixels = pixels * keep[:, :, None]
        pixels = np.concatenate([pixels, np.ones((height, width, 1), dtype=np.float32)], axis=2)
        return self.vae.encode(mx.array(pixels).transpose(2, 0, 1)[None])

    @staticmethod
    def _open(image: str | Path | Image.Image | None, mode: str = "RGB") -> Image.Image | None:
        return None if image is None else open_oriented(image).convert(mode)

    @staticmethod
    def _recorded(image: str | Path | Image.Image | None) -> str | None:
        # What the metadata records: the path, or a placeholder for an image passed in memory.
        if image is None:
            return None
        return str(image) if isinstance(image, (str, Path)) else "<in-memory image>"
