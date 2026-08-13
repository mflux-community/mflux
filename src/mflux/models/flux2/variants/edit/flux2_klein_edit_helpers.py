import math
from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.image_util import ImageUtil

# Reference images are conditioned at (at most) this many pixels, like the
# diffusers Flux2KleinPipeline. Larger references are downscaled (aspect-preserving)
# to fit; they are never resized to the output dimensions.
MAX_REFERENCE_AREA = 1024 * 1024

# Reference dims must be a multiple of 16 pixels (vae_scale_factor 8 * patch size 2).
REFERENCE_DIM_MULTIPLE = 16


class _Flux2KleinEditHelpers:
    @staticmethod
    def encode_text(
        prompt: str,
        *,
        tokenizer,
        text_encoder: Qwen3TextEncoder,
    ) -> tuple[mx.array, mx.array]:
        return Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

    @staticmethod
    def latent_grid_from_image_size(height: int, width: int) -> tuple[int, int]:
        vae_scale_factor = 8
        effective_height = 2 * (height // (vae_scale_factor * 2))
        effective_width = 2 * (width // (vae_scale_factor * 2))
        latent_height = effective_height // 2
        latent_width = effective_width // 2
        return latent_height, latent_width

    @staticmethod
    def build_latent_ids_grid(batch_size: int, latent_height: int, latent_width: int) -> mx.array:
        h_ids = mx.arange(latent_height, dtype=mx.int32)
        w_ids = mx.arange(latent_width, dtype=mx.int32)
        h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (latent_height, latent_width))
        w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (latent_height, latent_width))

        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = mx.zeros_like(flat_h)
        layer_ids = mx.zeros_like(flat_h)

        coords = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
        coords = mx.expand_dims(coords, axis=0)
        return mx.broadcast_to(coords, (batch_size, coords.shape[1], coords.shape[2]))

    @staticmethod
    def prepare_generation_latents(
        *,
        seed: int,
        height: int,
        width: int,
    ) -> tuple[mx.array, mx.array, int, int]:
        return Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=height,
            width=width,
            batch_size=1,
        )

    @staticmethod
    def crop_to_even_spatial(latents: mx.array) -> mx.array:
        if latents.shape[2] % 2 != 0:
            latents = latents[:, :, :-1, :]
        if latents.shape[3] % 2 != 0:
            latents = latents[:, :, :, :-1]
        return latents

    @staticmethod
    def ensure_4d_latents(latents: mx.array) -> mx.array:
        if latents.ndim == 5 and latents.shape[2] == 1:
            return latents[:, :, 0, :, :]
        return latents

    @staticmethod
    def bn_normalize_vae_encoded_latents(encoded: mx.array, *, vae: Flux2VAE) -> mx.array:
        bn_mean = vae.bn.running_mean.reshape(1, -1, 1, 1).astype(encoded.dtype)
        bn_std = mx.sqrt(vae.bn.running_var.reshape(1, -1, 1, 1) + vae.bn.eps).astype(encoded.dtype)
        return (encoded - bn_mean) / bn_std

    @staticmethod
    def reference_dims(width: int, height: int) -> tuple[int, int]:
        # 1. Aspect-preserving downscale so the area is at most MAX_REFERENCE_AREA
        area = width * height
        if area > MAX_REFERENCE_AREA:
            scale = math.sqrt(MAX_REFERENCE_AREA / area)
            width = round(width * scale)
            height = round(height * scale)

        # 2. Snap each dim down to a multiple of 16 (realized as a center-crop, never a squash)
        width -= width % REFERENCE_DIM_MULTIPLE
        height -= height % REFERENCE_DIM_MULTIPLE
        if width == 0 or height == 0:
            raise ValueError(f"Reference image too small to condition on (needs >= {REFERENCE_DIM_MULTIPLE}px per side)")  # fmt: off
        return width, height

    @staticmethod
    def prepare_reference_image(image: PIL.Image.Image) -> PIL.Image.Image:
        # Each reference keeps its OWN aspect ratio, independent of the output dims:
        # aspect-preserving downscale to at most MAX_REFERENCE_AREA, then center-crop
        # the sub-16px residual so each dim is a multiple of 16.
        target_width, target_height = _Flux2KleinEditHelpers.reference_dims(image.width, image.height)

        area = image.width * image.height
        if area > MAX_REFERENCE_AREA:
            scale = math.sqrt(MAX_REFERENCE_AREA / area)
            image = image.resize((round(image.width * scale), round(image.height * scale)), PIL.Image.LANCZOS)

        if (image.width, image.height) != (target_width, target_height):
            left = (image.width - target_width) // 2
            top = (image.height - target_height) // 2
            image = image.crop((left, top, left + target_width, top + target_height))
        return image

    @staticmethod
    def prepare_reference_image_conditioning(
        *,
        vae: Flux2VAE,
        tiling_config,
        image_paths: list[Path | str] | None = None,
        batch_size: int = 1,
    ):
        if not image_paths:
            return None, None

        packed_latents_list: list[mx.array] = []
        ids_list: list[mx.array] = []
        for i, p in enumerate(image_paths):
            image = _Flux2KleinEditHelpers.prepare_reference_image(ImageUtil.load_image(p).convert("RGB"))
            encoded = VAEUtil.encode(
                vae=vae,
                image=ImageUtil.to_array(image),
                tiling_config=tiling_config,
            )
            encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
            encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
            encoded = Flux2LatentCreator.patchify_latents(encoded)
            encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=vae)

            packed_latents_list.append(Flux2LatentCreator.pack_latents(encoded))
            ids_list.append(Flux2LatentCreator.prepare_grid_ids(encoded, t_coord=10 + 10 * i))

        image_latents = mx.concatenate(packed_latents_list, axis=1)
        image_latent_ids = mx.concatenate(ids_list, axis=1)

        if image_latents.shape[0] != batch_size:
            image_latents = mx.broadcast_to(image_latents, (batch_size, image_latents.shape[1], image_latents.shape[2]))
        if image_latent_ids.shape[0] != batch_size:
            image_latent_ids = mx.broadcast_to(
                image_latent_ids, (batch_size, image_latent_ids.shape[1], image_latent_ids.shape[2])
            )

        return image_latents, image_latent_ids
