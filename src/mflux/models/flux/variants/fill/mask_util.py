from pathlib import Path

import mlx.core as mx

from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.utils.image_util import ImageUtil


class MaskUtil:
    @staticmethod
    def create_masked_latents(
        vae: VAE,
        height: int,
        width: int,
        img_path: str | Path,
        mask_path: str | Path | None,
    ) -> mx.array:
        if not img_path or not mask_path:
            # Return empty latents if no image or mask is provided
            return mx.zeros((1, 0, 0))

        # 1. Get the reference image
        scaled_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(img_path).convert("RGB"),
            target_width=width,
            target_height=height,
        )
        image = ImageUtil.to_array(scaled_image)

        # 2. Get the mask
        scaled = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(mask_path, composite_alpha=False).convert("RGB"),
            target_width=width,
            target_height=height,
        )
        the_mask = ImageUtil.to_array(scaled, is_mask=True)

        # 3. Create and pack the masked image
        masked_image = image * (1 - the_mask)
        masked_image = vae.encode(masked_image)
        masked_image = FluxLatentCreator.pack_latents(latents=masked_image, height=height, width=width)

        # 4. Resize mask and pack latents
        mask = MaskUtil.reshape_mask(the_mask=the_mask, height=height, width=width)
        mask = FluxLatentCreator.pack_latents(latents=mask, height=height, width=width, num_channels_latents=64)

        # 5. Concat the masked_image and the mask
        masked_image_latents = mx.concatenate([masked_image, mask], axis=-1)
        return masked_image_latents

    @staticmethod
    def reshape_mask(the_mask: mx.array, height: int, width: int) -> mx.array:
        mask = the_mask[:, 0, :, :]
        mask = mx.reshape(mask, (1, height // 8, 8, width // 8, 8))
        mask = mx.transpose(mask, (0, 2, 4, 1, 3))
        mask = mx.reshape(mask, (1, 64, height // 8, width // 8))
        return mask
