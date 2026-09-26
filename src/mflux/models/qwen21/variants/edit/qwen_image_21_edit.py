import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_grounding import Qwen21Grounding
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer, StepCache
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.qwen21_edit_initializer import Qwen21EditInitializer
from mflux.models.qwen21.tokenizer.qwen21_image_processor import Qwen21ImageProcessor
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.exif_orientation import open_oriented
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

logger = logging.getLogger(__name__)


class QwenImage21Edit(nn.Module):
    vae: Qwen21VAE
    transformer: Qwen21Transformer
    text_encoder: Qwen21TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.qwen_image_21(),
    ):
        super().__init__()
        Qwen21EditInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    @staticmethod
    def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
        # Port of the reference pipeline's calculate_dimensions (multiples of 32).
        width = (target_area * ratio) ** 0.5
        height = width / ratio
        return round(width / 32) * 32, round(height / 32) * 32

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: list[str] | list[Path] | list[Image.Image],
        num_inference_steps: int = 40,
        height: int | None = None,
        width: int | None = None,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        scheduler: str = "linear",
        output_resolution: int = 1024,
        use_kv_cache: bool = True,
        mask_image: str | Path | Image.Image | None = None,
        auto_mask: str | None = None,
        use_step_cache: bool = False,
        step_cache_threshold: float = 0.12,
        strength: float = 1.0,
        enhance_prompt: bool = False,
        verify: bool = False,
        verify_retries: int = 0,
        rgba_output: bool = False,
    ) -> GeneratedImage:
        # Normalize inputs to PIL up front (same normalization order as the reference).
        # open_oriented applies the file's EXIF Orientation tag so a portrait JPEG stored
        # landscape is encoded the way it displays; RGBA mode is preserved for the VAE.
        images = [open_oriented(img if isinstance(img, Image.Image) else img) for img in image_paths]

        if not 0.0 < strength <= 1.0:
            raise ValueError(f"strength must be in (0, 1], got {strength}")
        if len(images) > 10:
            raise ValueError(f"Qwen-Image-2.1 supports at most 10 condition images, got {len(images)}")

        # 0. Optional prompt rewriting: the official Qwen-Image-2.1 serving recipe
        # rewrites terse edit instructions into detailed descriptions with a VL model
        # before encoding. The in-memory Qwen3-VL (with its untied lm_head) does it
        # without extra weights; the ORIGINAL instruction still drives verification.
        original_prompt = prompt
        if enhance_prompt:
            prompt = self._rewrite_prompt(prompt, vision_source=images[0])
            logger.info("enhance_prompt: %r -> %r", original_prompt, prompt)

        # 1. Resize every condition image to its own area-normalized, /32-aligned size.
        # The vision encoder rejects sequences beyond a 200:1 aspect ratio; the bounds
        # are checked here, on the rounded dimensions that actually reach it, so an
        # extreme panorama fails with a clear input error instead of a ValueError from
        # deep inside preprocessing (a ratio so extreme also rounds a side to 0).
        resized_images: list[Image.Image] = []
        ref_shapes: list[tuple[int, int]] = []  # latent-grid (h, w) per image
        for img in images:
            ratio = img.size[0] / img.size[1]
            w32, h32 = self._calculate_dimensions(output_resolution * output_resolution, ratio)
            if min(w32, h32) < 32 or max(w32, h32) / min(w32, h32) > 200:
                raise ValueError(
                    f"condition image {img.size} (aspect ratio {ratio:.1f}) is outside the "
                    f"supported 200:1 range once rounded to /32 multiples ({w32}x{h32})"
                )
            resized = img.resize((w32, h32), Image.BICUBIC) if img.size != (w32, h32) else img
            resized_images.append(resized)
            ref_shapes.append((h32 // 16, w32 // 16))

        # Output size: an explicitly passed axis is honored, a missing one derives from
        # the last condition image's aspect ratio (per-axis, like the reference pipeline).
        width = width if width is not None else ref_shapes[-1][1] * 16
        height = height if height is not None else ref_shapes[-1][0] * 16

        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            # reuse the img2img machinery for edit strength: image_path + strength make
            # init_time_step start the denoising loop partway down the sigma schedule
            image_path=str(image_paths[0]) if not isinstance(image_paths[0], Image.Image) else "condition",
            image_strength=strength if strength < 1.0 else None,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )
        latents = Qwen21LatentCreator.create_noise(seed=seed, height=config.height, width=config.width)
        latents = latents.astype(ModelConfig.precision)

        # 2. Vision path: RGBA is composited over white for the vision encoder only.
        vision_images = []
        for img in resized_images:
            if img.mode == "RGBA":
                white = Image.new("RGB", img.size, (255, 255, 255))
                white.paste(img, mask=img.getchannel("A"))
                vision_images.append(white)
            else:
                vision_images.append(img.convert("RGB"))
        pixel_values, grid_thw = Qwen21ImageProcessor().preprocess(vision_images)

        # 2b. Inpaint mask: an explicit mask_image wins; auto_mask instead asks the
        # in-memory Qwen3-VL where the named object is and rasterizes its answer.
        # White = repaint, black = preserve; the mask drives per-step latent blending
        # plus a final pixel composite.
        inpaint_mask = self._resolve_mask(
            mask_image=mask_image,
            auto_mask=auto_mask,
            vision_image=vision_images[0],
            width=width,
            height=height,
        )

        # 3. Pixel path: the VAE encodes full RGBA (the alpha channel can carry edit masks).
        ref_latents = []
        for img in resized_images:
            vae_image = QwenImage21Edit._to_vae_tensor(img)
            encoded = self.vae.encode(vae_image)  # (1, 64, h, w), latents-normalized
            ref_latents.append(
                Qwen21LatentCreator.pack_latents(
                    latents=encoded,
                    height=encoded.shape[2] * 16,
                    width=encoded.shape[3] * 16,
                )
            )
        ref_latents = mx.concatenate(ref_latents, axis=1).astype(ModelConfig.precision)

        # 3b. Reference at output dimensions: the inpaint blend source, and the anchor
        # for partial-strength starts. x_sigma = ref + sigma * (noise - ref) is the
        # flow-matching interpolation that matches the Euler update, so unmasked tokens
        # can be pulled onto the reference's own trajectory each step, and strength < 1
        # can start from the reference noised at the schedule's starting sigma.
        start_step = config.init_time_step  # 0 unless strength < 1
        blend_mask = None
        blend_source = None
        init_noise = None
        blend_image = None
        if inpaint_mask is not None or start_step > 0:
            latent_h, latent_w = config.height // 16, config.width // 16
            base = images[0].convert("RGBA")
            white = Image.new("RGB", base.size, (255, 255, 255))
            white.paste(base, mask=base.getchannel("A"))
            blend_image = white.resize((config.width, config.height), Image.BICUBIC)
            encoded = self.vae.encode(QwenImage21Edit._to_vae_tensor(blend_image))
            blend_source = Qwen21LatentCreator.pack_latents(
                latents=encoded, height=encoded.shape[2] * 16, width=encoded.shape[3] * 16
            ).astype(ModelConfig.precision)
            init_noise = latents
        if inpaint_mask is not None:
            mask_grid = Qwen21Grounding.to_latent_mask(inpaint_mask, latent_h, latent_w)
            blend_mask = mx.array(mask_grid.reshape(1, latent_h * latent_w, 1)).astype(ModelConfig.precision)
        if start_step > 0:
            # strength start: seed the loop from the reference noised to sigma_start
            # instead of pure noise; the loop below then begins at step `start_step`
            sigma_start = config.scheduler.sigmas[start_step].astype(latents.dtype)
            latents = blend_source + sigma_start * (init_noise - blend_source)

        # 4. Encode prompt + condition images with the Qwen3-VL encoder into the
        # template-ordered run layout the transformer consumes.
        prompt_layout = self._encode_prompt_with_images(
            prompt=prompt,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            ref_latents=ref_latents,
            ref_shapes=ref_shapes,
            tokenizer=self.tokenizers["qwen21"],
            text_encoder=self.text_encoder,
        )
        negative_prompt_layout = None
        if config.guidance > 1.0 and not negative_prompt:
            logger.warning(
                f"guidance={config.guidance} has no effect without a negative prompt; "
                "pass negative_prompt to enable classifier-free guidance"
            )
        do_true_cfg = config.guidance > 1.0 and bool(negative_prompt)
        if do_true_cfg:
            negative_prompt_layout = self._encode_prompt_with_images(
                prompt=negative_prompt,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                ref_latents=ref_latents,
                ref_shapes=ref_shapes,
                tokenizer=self.tokenizers["qwen21"],
                text_encoder=self.text_encoder,
            )

        # 5. Denoising loop over the reference joint sequences. The prefix KV cache is
        # valid because causal_condition keeps text/reference activations step-independent:
        # the first step prefills, later steps recompute only the target queries. The
        # conditional and unconditional passes need separate caches (different embeds).
        kv_cache = [None] * len(self.transformer.transformer_blocks) if use_kv_cache else None
        neg_kv_cache = [None] * len(self.transformer.transformer_blocks) if (use_kv_cache and do_true_cfg) else None
        step_cache = StepCache(step_cache_threshold) if use_step_cache else None
        neg_step_cache = StepCache(step_cache_threshold) if (use_step_cache and do_true_cfg) else None
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for step, t in enumerate(config.time_steps):
            try:
                latents = config.scheduler.scale_model_input(latents, t)
                kv_mode = "extract" if step == 0 else "cached"
                noise = self.transformer.__call_edit__(
                    t=t,
                    config=config,
                    target_latents=latents,
                    layout=prompt_layout,
                    kv_cache=kv_cache,
                    kv_cache_mode=kv_mode if use_kv_cache else None,
                    step_cache=step_cache,
                )
                if do_true_cfg:
                    noise_negative = self.transformer.__call_edit__(
                        t=t,
                        config=config,
                        target_latents=latents,
                        layout=negative_prompt_layout,
                        kv_cache=neg_kv_cache,
                        kv_cache_mode=kv_mode if use_kv_cache else None,
                        step_cache=neg_step_cache,
                    )
                    noise = noise_negative + config.guidance * (noise - noise_negative)

                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                if blend_mask is not None:
                    # after the Euler update the state sits at sigma_{t+1}; pull the
                    # unmasked tokens onto the reference's own trajectory at that sigma
                    sigma_next = config.scheduler.sigmas[t + 1].astype(latents.dtype)
                    noised_ref = blend_source + sigma_next * (init_noise - blend_source)
                    latents = blend_mask * latents + (1.0 - blend_mask) * noised_ref
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        latents = Qwen21LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config, keep_alpha=rgba_output)
        if inpaint_mask is not None:
            keep = np.asarray(inpaint_mask, dtype=np.float32) / 255.0  # (H, W): 1 = repaint
            original = np.asarray(blend_image, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0
            if rgba_output and original.shape[1] == 3:
                # composite over the same channel count the decoder produced
                original = np.concatenate([original, np.ones_like(original[:, :1])], axis=1)
            repaint = mx.array(keep[None, None, :, :])
            original = mx.array(original)
            # float32 compositing keeps unmasked pixels exactly the original
            decoded = repaint * decoded.astype(mx.float32) + (1.0 - repaint) * original

        image = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
            image_paths=image_paths if not isinstance(image_paths[0], Image.Image) else None,
        )
        if not verify:
            return image

        # 6. Optional self-check with the in-memory Qwen3-VL; on a failed verdict,
        # retry with a different seed (the prompt/layout work is cached per call only,
        # so each retry pays a full generation -- hence retries are opt-in).
        verification = self._verify_output(original_prompt, images[0], image.image)
        retries = 0
        while not verification.get("verified") and retries < verify_retries:
            retries += 1
            logger.info("verify failed (%s); retry %d/%d", verification, retries, verify_retries)
            retry_image = self.generate_image(
                seed=seed + retries,
                prompt=original_prompt,
                image_paths=image_paths,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance=guidance,
                negative_prompt=negative_prompt,
                scheduler=scheduler,
                output_resolution=output_resolution,
                use_kv_cache=use_kv_cache,
                mask_image=mask_image,
                auto_mask=auto_mask,
                use_step_cache=use_step_cache,
                step_cache_threshold=step_cache_threshold,
                strength=strength,
                enhance_prompt=enhance_prompt,
            )
            retry_verification = self._verify_output(original_prompt, images[0], retry_image.image)
            if retry_verification.get("verified"):
                logger.info("verify passed on retry %d", retries)
                retry_image.verification = {**retry_verification, "retries": retries}
                return retry_image
        image.verification = {**verification, "retries": retries}
        return image

    def _resolve_mask(
        self,
        mask_image: str | Path | Image.Image | None,
        auto_mask: str | None,
        vision_image: Image.Image,
        width: int,
        height: int,
    ) -> Image.Image | None:
        if mask_image is not None and auto_mask is not None:
            logger.warning("both mask_image and auto_mask were given; using the explicit mask_image")
        if mask_image is not None:
            return open_oriented(mask_image).convert("L").resize((width, height), Image.BILINEAR)
        if auto_mask is None:
            return None
        if not auto_mask.strip():
            raise ValueError("auto_mask must name an object to locate")
        # Ground on a small copy: fewer image tokens make the prefill cheap. The reply's
        # coordinates are relative to the shown image, so its size is kept for parsing.
        ratio = vision_image.size[0] / vision_image.size[1]
        feed_width, feed_height = self._calculate_dimensions(512 * 512, ratio)
        grounding_image = vision_image.resize((feed_width, feed_height), Image.BICUBIC)
        pixel_values, grid_thw = Qwen21ImageProcessor().preprocess([grounding_image])
        _, grid_h, grid_w = (int(v) for v in grid_thw[0].tolist())
        n_tokens = (grid_h // 2) * (grid_w // 2)

        tokenizer = self.tokenizers["qwen21"]
        input_ids = mx.array(
            [Qwen21Grounding.build_input_ids(tokenizer, n_tokens, auto_mask)],
            dtype=mx.int32,
        )
        reply_ids = self.text_encoder.locate_object(input_ids, pixel_values, grid_thw)
        reply = tokenizer.tokenizer.decode(reply_ids)
        bbox = Qwen21Grounding.parse_bbox(reply, (feed_width, feed_height))
        if bbox is None:
            raise ValueError(
                f"auto_mask could not locate '{auto_mask}' in the first condition image "
                f"(model reply: {reply[:120]!r}); provide an explicit mask_image instead"
            )
        logger.info("auto_mask '%s' resolved to bbox %s", auto_mask, tuple(round(v, 3) for v in bbox))
        return Qwen21Grounding.rasterize_mask(bbox, (width, height))

    def _rewrite_prompt(self, prompt: str, vision_source: Image.Image) -> str:
        # Official serving recipe: rewrite a terse instruction into a detailed
        # description before encoding. Grounded on a small copy of the first condition
        # image; a failed parse falls back to the original instruction (never worse).
        if not prompt.strip():
            return prompt
        try:
            feed = self._vision_feed(vision_source)
            pixel_values, grid_thw = feed
            _, grid_h, grid_w = (int(v) for v in grid_thw[0].tolist())
            n_tokens = (grid_h // 2) * (grid_w // 2)
            tokenizer = self.tokenizers["qwen21"]
            text = (
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                + Qwen21Grounding.REWRITE_PROMPT.format(instruction=prompt)
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            input_ids = mx.array([Qwen21Grounding.tokenize_with_images(tokenizer, text, [n_tokens])], dtype=mx.int32)
            reply_ids = self.text_encoder.generate(
                input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw, max_new_tokens=256
            )
            reply = tokenizer.tokenizer.decode(reply_ids)
            rewritten = Qwen21Grounding.parse_rewrite(reply)
            if rewritten is None:
                logger.warning("enhance_prompt could not parse the rewrite reply; using the original")
                return prompt
            return rewritten
        except Exception as exc:  # noqa: BLE001 - rewriting is best-effort by design
            logger.warning("enhance_prompt failed (%s); using the original", exc)
            return prompt

    def _vision_feed(self, vision_image: Image.Image, budget: int = 512) -> tuple[mx.array, mx.array]:
        ratio = vision_image.size[0] / vision_image.size[1]
        feed_width, feed_height = self._calculate_dimensions(budget * budget, ratio)
        grounding_image = vision_image.resize((feed_width, feed_height), Image.BICUBIC)
        return Qwen21ImageProcessor().preprocess([grounding_image])

    def _verify_output(self, instruction: str, original: Image.Image, output: Image.Image) -> dict:
        # Post-edit self-check with the in-memory Qwen3-VL: original + output side by
        # side, structured verdict parsed from the reply. Coarse by design (did the
        # edit apply; is the rest preserved) -- not an aesthetic score.
        original_rgb = original.convert("RGB")
        output_rgb = output.convert("RGB")
        feeds = [self._vision_feed(original_rgb), self._vision_feed(output_rgb)]
        pixel_values = mx.concatenate([pv for pv, _ in feeds], axis=0)
        grid_thw = mx.concatenate([g for _, g in feeds], axis=0)
        token_counts = []
        for grid in (feeds[0][1], feeds[1][1]):
            _, grid_h, grid_w = (int(v) for v in grid[0].tolist())
            token_counts.append((grid_h // 2) * (grid_w // 2))
        tokenizer = self.tokenizers["qwen21"]
        text = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "<|vision_start|><|image_pad|><|vision_end|>"
            + Qwen21Grounding.VERIFY_PROMPT.format(instruction=instruction)
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids = mx.array([Qwen21Grounding.tokenize_with_images(tokenizer, text, token_counts)], dtype=mx.int32)
        reply_ids = self.text_encoder.generate(
            input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw, max_new_tokens=64
        )
        reply = tokenizer.tokenizer.decode(reply_ids)
        verdict = Qwen21Grounding.parse_verification(reply)
        if verdict is None:
            return {"verified": False, "parse_failed": True, "reply": reply[:120]}
        applied, unchanged = verdict
        return {
            "verified": applied and unchanged,
            "instruction_applied": applied,
            "outside_unchanged": unchanged,
        }

    def _encode_prompt_with_images(
        self,
        prompt: str,
        pixel_values: mx.array,
        grid_thw: mx.array,
        ref_latents: mx.array,
        ref_shapes: list[tuple[int, int]],
        tokenizer,
        text_encoder: Qwen21TextEncoder,
    ) -> list[tuple]:
        # Returns the template-ordered run layout: ('text', embeds) / ('image', latents,
        # (h, w)) runs, with the target block added by the transformer. The ti2i template
        # is '<system><|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>
        # [prompt]<|im_end|>\n<|im_start|>assistant' with each <|image_pad|> expanded to
        # the image's merged token count. After dropping the system prefix, the hidden
        # states interleave [user header][image slots][prompt tail]; the layout splits
        # them at the image-slot span and splices the reference latent blocks in place of
        # the slots -- sequence-position equivalent to the reference's slot expansion.
        if not prompt or not prompt.strip():
            prompt = " "

        slot_runs = []
        for i in range(len(ref_shapes)):
            _, gh, gw = (int(v) for v in grid_thw[i].tolist())
            n_tokens = (gh // 2) * (gw // 2)
            prefix = "<image1>" if i == 0 else f" <image{i + 1}>"
            slot_runs.append(f"{prefix}<|vision_start|>{'<|image_pad|>' * n_tokens}<|vision_end|>")
        template = (
            f"<|im_start|>system\n{Qwen21PromptEncoder.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{''.join(slot_runs)}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )

        tokens = tokenizer.tokenizer(template, add_special_tokens=False, return_tensors="np")
        input_ids = mx.array(np.asarray(tokens["input_ids"])).astype(mx.int32)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        hidden_states, image_mask = text_encoder.forward_vl(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
        )

        drop_idx = len(
            tokenizer.tokenizer(
                f"<|im_start|>system\n{Qwen21PromptEncoder.SYSTEM_PROMPT}<|im_end|>\n",
                add_special_tokens=False,
            )["input_ids"]
        )
        post_mask = np.array(image_mask[0])[drop_idx:]
        post_hidden = hidden_states[0][drop_idx:]

        # Split the post-drop sequence into runs: image slots consumed per ref shape,
        # everything between them grouped into text runs.
        layout: list[tuple] = []
        cursor = 0  # index into post-drop VLM sequence positions (merged slots)
        latent_cursor = 0  # index into the packed reference latents (4 tokens per slot)
        image_index = 0
        while cursor < len(post_mask):
            if post_mask[cursor]:
                h, w = ref_shapes[image_index]
                n_slots = (h // 2) * (w // 2)  # merged slots the VLM sequence reserves
                n_latent = h * w  # latent tokens substituted on expansion (4 per slot)
                if not post_mask[cursor : cursor + n_slots].all():
                    raise ValueError("image slot run is not contiguous in the tokenized template")
                image_index += 1
                layout.append(("image", ref_latents[:, latent_cursor : latent_cursor + n_latent], (h, w)))
                latent_cursor += n_latent
                cursor += n_slots
            else:
                window = post_mask[cursor:]
                true_positions = np.flatnonzero(window)
                next_image = cursor + (int(true_positions[0]) if len(true_positions) else len(window))
                layout.append(("text", post_hidden[None, cursor:next_image, :].astype(ModelConfig.precision)))
                cursor = next_image
        if image_index != len(ref_shapes):
            raise ValueError(
                f"image slot runs consumed {image_index} images but {len(ref_shapes)} condition images were given"
            )
        return layout

    @staticmethod
    def _to_vae_tensor(image: Image.Image) -> mx.array:
        # (1, 4, H, W) in [-1, 1]; RGBA keeps its alpha channel (edit masks).
        array = np.array(image.convert("RGBA")).astype(np.float32) / 127.5 - 1.0
        array = array.transpose(2, 0, 1)[None]
        return mx.array(array)
