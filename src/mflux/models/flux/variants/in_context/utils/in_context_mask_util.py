import mlx.core as mx

from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.flux.variants.fill.mask_util import MaskUtil
from mflux.utils.image_util import ImageUtil


class InContextMaskUtil:
    @staticmethod
    def create_masked_latents(
        vae: VAE,
        height: int,
        width: int,
        original_width: int,
        left_image_path: str,
        right_image_path: str | None,
        mask_path: str,
    ):
        # Determine mode based on whether right_image_path is provided
        is_pure_mode = right_image_path is None

        # Step 1: Load left image (reference) - always required
        left_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(left_image_path).convert("RGB"),
            target_width=original_width,
            target_height=height,
        )
        left_array = ImageUtil.to_array(left_image)

        # Step 2: Handle right image based on mode
        if is_pure_mode:
            # Pure mode: create empty/noise for right side
            right_array = mx.zeros_like(left_array)  # or could use noise
        else:
            # Selective mode: load target image for right side
            right_image = ImageUtil.scale_to_dimensions(
                image=ImageUtil.load_image(right_image_path).convert("RGB"),
                target_width=original_width,
                target_height=height,
            )
            right_array = ImageUtil.to_array(right_image)

        # Step 3: Determine the right-side mask based on mode
        if is_pure_mode:
            # Pure in-context: generate entire right side
            mask_array = mx.ones((1, 1, height, original_width))
        else:
            # Selective: use provided mask exactly
            mask_image = ImageUtil.scale_to_dimensions(
                image=ImageUtil.load_image(mask_path, composite_alpha=False).convert("RGB"),
                target_width=original_width,
                target_height=height,
            )
            mask_array = ImageUtil.to_array(mask_image, is_mask=True)

        # Step 4: Create concatenated inputs for in-context learning
        # Layout: [LEFT_IMAGE] | [RIGHT_IMAGE]
        concatenated_image = mx.concatenate([left_array, right_array], axis=3)

        # Layout: [NO_MASK] | [ACTUAL_MASK]
        reference_mask = mx.zeros_like(mask_array)  # Empty mask for reference (preserve fully)
        concatenated_mask = mx.concatenate([reference_mask, mask_array], axis=3)

        # Step 5: Apply standard FLUX Fill processing to concatenated inputs
        masked_concatenated_image = concatenated_image * (1 - concatenated_mask)

        # Encode and process exactly like MaskUtil.create_masked_latents
        encoded_image = vae.encode(masked_concatenated_image)
        encoded_image = FluxLatentCreator.pack_latents(
            latents=encoded_image,
            height=height,
            width=width,
        )

        processed_mask = MaskUtil.reshape_mask(
            the_mask=concatenated_mask,
            height=height,
            width=width,
        )
        processed_mask = FluxLatentCreator.pack_latents(
            latents=processed_mask,
            height=height,
            width=width,
            num_channels_latents=64,
        )

        return mx.concatenate([encoded_image, processed_mask], axis=-1)
