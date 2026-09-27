import hashlib
import json
import logging
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
from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.grounding import QwenImage21Grounding
from mflux.models.qwen21.model.qwen21_text_encoder.prompt_encoder import QwenImage21PromptEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import StepCache
from mflux.models.qwen21.qwen_image21_initializer import QwenImage21Initializer
from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.exif_orientation import open_oriented
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

logger = logging.getLogger(__name__)


class QwenImage21Edit(nn.Module):
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
        QwenImage21Initializer.init(
            self, model_config or ModelConfig.qwen_image_21(), quantize, model_path, lora_paths, lora_scales, bake_lora
        )

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
        mask_image: str | Path | Image.Image | None = None,
        auto_mask: str | None = None,
        strength: float = 1.0,
        use_step_cache: bool = False,
        step_cache_threshold: float = 0.12,
        enhance_prompt: bool = False,
        verify: bool = False,
        verify_retries: int = 0,
    ) -> GeneratedImage:
        image_paths = image_paths or []
        if len(image_paths) > 10:
            raise ValueError("Qwen-Image-2.1 supports at most 10 reference images.")
        QwenImage21LatentCreator.validate_resolution(output_resolution)
        if not 0 < strength <= 1:
            raise ValueError(f"strength must be in (0, 1], got {strength}.")
        if verify_retries < 0:
            raise ValueError(f"verify_retries must be >= 0, got {verify_retries}.")
        needs_source = mask_image is not None or auto_mask is not None or strength < 1 or enhance_prompt or verify
        if needs_source and not image_paths:
            raise ValueError("mask_image, auto_mask, strength, enhance_prompt and verify need a reference image.")
        images = [open_oriented(path).convert("RGBA") for path in image_paths]
        # The first reference is the edit source: inpaint blending, strength starts,
        # prompt rewriting and verification all read it at its own size.
        source = images[0] if images else None
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
            # strength < 1 skips the first (1 - strength) of the schedule, starting from
            # the source noised to that sigma instead of pure noise
            image_path=str(image_paths[0]) if strength < 1 else None,
            image_strength=1 - strength if strength < 1 else None,
        )
        original_prompt = prompt
        if enhance_prompt:
            prompt = self._rewrite_prompt(prompt, source)
            logger.info("enhance_prompt: %r -> %r", original_prompt, prompt)
        # White = repaint, black = preserve; drives per-step latent blending and a final pixel composite.
        inpaint_mask = self._resolve_mask(mask_image, auto_mask, source, width, height)
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
        # Flow matching interpolates x_sigma = src + sigma * (noise - src), so unmasked
        # tokens can follow the source's own trajectory and strength < 1 can start on it.
        noise_init = latents
        blend_image = blend_source = blend_mask = None
        if inpaint_mask is not None or config.init_time_step > 0:
            blend_image = source.resize((width, height), Image.Resampling.LANCZOS)
            pixels = mx.array(np.asarray(blend_image).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
            blend_source = QwenImage21LatentCreator.pack_latents(self.vae.encode(pixels)).astype(latents.dtype)
        if inpaint_mask is not None:
            grid = QwenImage21Grounding.to_latent_mask(inpaint_mask, height // 16, width // 16)
            blend_mask = mx.array(grid.reshape(1, -1, 1)).astype(latents.dtype)
        if config.init_time_step > 0:
            sigma_start = config.scheduler.sigmas[config.init_time_step]
            latents = (blend_source + sigma_start * (noise_init - blend_source)).astype(latents.dtype)
        cache = [] if use_kv_cache else None
        negative_cache = [] if use_kv_cache else None
        if use_step_cache and not use_kv_cache:
            logger.warning("use_step_cache needs use_kv_cache; running without step skipping")
        step_cache = StepCache(step_cache_threshold) if use_step_cache and use_kv_cache else None
        negative_step_cache = (
            StepCache(step_cache_threshold) if step_cache is not None and negative is not None else None
        )
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                model_input = mx.concatenate([condition_latents, latents], axis=1) if conditions else latents
                # Match diffusers: cast the 0..1000 timestep before dividing by 1000.
                timestep = (config.scheduler.sigmas[t : t + 1] * 1000).astype(latents.dtype) / 1000
                noise = self.transformer(model_input, prompt_embeds, timestep, layout, cache, step_cache=step_cache)
                if negative is not None:
                    uncond = self.transformer(
                        model_input,
                        negative[0],
                        timestep,
                        negative_layout,
                        negative_cache,
                        step_cache=negative_step_cache,
                    )
                    noise = uncond + guidance * (noise - uncond)
                # The upstream Euler update accumulates in fp32 and casts back afterward.
                sigma = config.scheduler.sigmas
                latents = (latents.astype(mx.float32) + (sigma[t + 1] - sigma[t]) * noise.astype(mx.float32)).astype(
                    latents.dtype
                )
                if blend_mask is not None:
                    # the state now sits at sigma_{t+1}; pin unmasked tokens to the source there
                    noised = blend_source + sigma[t + 1] * (noise_init - blend_source)
                    latents = (blend_mask * latents + (1 - blend_mask) * noised).astype(latents.dtype)
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
        if inpaint_mask is not None:
            # float32 compositing keeps unmasked pixels exactly the source
            keep = mx.array(np.asarray(inpaint_mask, dtype=np.float32) / 255.0)
            original = np.asarray(blend_image).astype(np.float32).transpose(2, 0, 1) / 127.5 - 1
            original = mx.array(original[: decoded.shape[1]]).reshape(decoded.shape)
            decoded = keep * decoded.astype(mx.float32) + (1 - keep) * original
        parameters = {"use_kv_cache": use_kv_cache, "output_resolution": output_resolution}
        if strength < 1:
            parameters["strength"] = strength
        if auto_mask is not None:
            parameters["auto_mask"] = auto_mask
        if use_step_cache:
            parameters["step_cache_threshold"] = step_cache_threshold
        if enhance_prompt:
            parameters["original_prompt"] = original_prompt
        image = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            generation_time=config.time_steps.format_dict["elapsed"],
            image_paths=image_paths,
            negative_prompt=negative_prompt,
            generation_parameters=parameters,
        )
        if not verify:
            return image
        # Coarse self-check with the in-memory Qwen3-VL; a failed verdict retries with the
        # next seed, each retry paying a full generation (hence opt-in). Retries reuse this
        # run's mask and rewritten prompt: both are deterministic per source.
        verification = self._verify_output(original_prompt, source, image.image)
        retries = 0
        while not verification.get("verified") and retries < verify_retries:
            retries += 1
            logger.info("verify failed (%s); retry %d/%d", verification, retries, verify_retries)
            retry_image = self.generate_image(
                seed=seed + retries,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance=guidance,
                negative_prompt=negative_prompt,
                image_paths=image_paths,
                output_resolution=output_resolution,
                use_kv_cache=use_kv_cache,
                mask_image=inpaint_mask,
                strength=strength,
                use_step_cache=use_step_cache,
                step_cache_threshold=step_cache_threshold,
            )
            # the retry got the resolved mask and prompt; record how they were derived
            retry_image.generation_parameters.update(
                {key: parameters[key] for key in ("auto_mask", "original_prompt") if key in parameters}
            )
            retry_verification = self._verify_output(original_prompt, source, retry_image.image)
            if retry_verification.get("verified"):
                retry_image.verification = {**retry_verification, "retries": retries}
                return retry_image
        image.verification = {**verification, "retries": retries}
        return image

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

    def _resolve_mask(
        self,
        mask_image: str | Path | Image.Image | None,
        auto_mask: str | None,
        source: Image.Image | None,
        width: int,
        height: int,
    ) -> Image.Image | None:
        if mask_image is not None and auto_mask is not None:
            logger.warning("both mask_image and auto_mask were given; using the explicit mask_image")
        if mask_image is not None:
            return open_oriented(mask_image).convert("L").resize((width, height), Image.Resampling.BILINEAR)
        if auto_mask is None:
            return None
        if not auto_mask.strip():
            raise ValueError("auto_mask must name an object to locate.")
        reply = self._vision_reply(QwenImage21Grounding.GROUNDING_PROMPT.format(query=auto_mask), [source], 64)
        bbox = QwenImage21Grounding.parse_bbox(reply)
        if bbox is None:
            raise ValueError(
                f"auto_mask could not locate {auto_mask!r} in the first reference image "
                f"(model reply: {reply[:120]!r}); provide mask_image instead."
            )
        logger.info("auto_mask %r resolved to bbox %s", auto_mask, tuple(round(v, 3) for v in bbox))
        return QwenImage21Grounding.rasterize_mask(bbox, (width, height))

    def _rewrite_prompt(self, prompt: str, source: Image.Image) -> str:
        # Official serving recipe: rewrite a terse instruction into a detailed description
        # before encoding. Best-effort: any failure falls back to the original instruction.
        if not prompt or not prompt.strip():
            return prompt
        try:
            reply = self._vision_reply(QwenImage21Grounding.REWRITE_PROMPT.format(instruction=prompt), [source], 384)
        except Exception as exc:  # noqa: BLE001
            logger.warning("enhance_prompt failed (%s); using the original", exc)
            return prompt
        rewritten = QwenImage21Grounding.parse_rewrite(reply)
        if rewritten is None:
            logger.warning("enhance_prompt could not parse the rewrite reply; using the original")
            return prompt
        return rewritten

    def _verify_output(self, instruction: str, original: Image.Image, output: Image.Image) -> dict:
        reply = self._vision_reply(
            QwenImage21Grounding.VERIFY_PROMPT.format(instruction=instruction), [original, output], 64
        )
        verdict = QwenImage21Grounding.parse_verification(reply)
        if verdict is None:
            return {"verified": False, "parse_failed": True, "reply": reply[:120]}
        applied, unchanged = verdict
        return {"verified": applied and unchanged, "instruction_applied": applied, "outside_unchanged": unchanged}

    def _vision_reply(self, instruction: str, images: list[Image.Image], max_new_tokens: int) -> str:
        # Greedy chat reply from the in-memory Qwen3-VL over small copies of the images.
        # Greedy decoding is deterministic, so a multi-seed run grounds and rewrites once
        # (the cache lives in __dict__, off the nn.Module parameter tree).
        digests = tuple(hashlib.sha1(image.tobytes()).hexdigest() + f"{image.size}{image.mode}" for image in images)
        key = (instruction, max_new_tokens, digests)
        cache = self.__dict__.setdefault("_vision_cache", {})
        if key in cache:
            return cache[key]
        if self.text_encoder is None:
            raise RuntimeError("The text encoder was released by the memory saver; reload the model.")
        feeds = [QwenImage21Edit._vision_feed(image) for image in images]
        text = QwenImage21Grounding.chat(instruction, len(feeds))
        inputs = self.processor(text=[text], images=feeds, return_tensors="np")
        ids = self.text_encoder.generate(
            mx.array(inputs["input_ids"]),
            pixel_values=mx.array(inputs["pixel_values"]),
            image_grid_thw=mx.array(inputs["image_grid_thw"]),
            max_new_tokens=max_new_tokens,
        )
        reply = self.processor.tokenizer.decode(ids)
        if len(cache) >= 32:
            cache.pop(next(iter(cache)))
        cache[key] = reply
        return reply

    @staticmethod
    def _vision_feed(image: Image.Image, budget: int = 512) -> Image.Image:
        # RGBA composites over white for the vision tower; ~512px keeps the prefill cheap.
        rgba = image.convert("RGBA")
        white = Image.new("RGB", rgba.size, "white")
        white.paste(rgba, mask=rgba.getchannel("A"))
        size = QwenImage21LatentCreator.dimensions(budget, image.width / image.height)
        return white.resize(size, Image.Resampling.BICUBIC)

    def _encode_prompt(self, prompt: str, images: list[Image.Image]) -> tuple[mx.array, mx.array]:
        if not images and prompt in self.prompt_cache:
            return self.prompt_cache[prompt]
        if self.text_encoder is None:
            raise RuntimeError("The text encoder was released by the memory saver; reload the model for a new prompt.")
        encoded = QwenImage21PromptEncoder.encode(prompt, images, self.processor, self.text_encoder)
        if not images:
            self.prompt_cache[prompt] = encoded
        return encoded
