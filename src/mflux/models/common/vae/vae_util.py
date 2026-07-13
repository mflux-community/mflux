import mlx.core as mx
from mlx import nn

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_tiler import VAETiler


class VAEUtil:
    @staticmethod
    def encode(
        vae: nn.Module,
        image: mx.array,
        tiling_config: TilingConfig | None = None,
    ) -> mx.array:
        # 1. Tiled encoding if enabled
        if tiling_config is not None and tiling_config.vae_encode_tiled:
            # VAETiler.encode_image_tiled always returns 5D (B, C, 1, H_lat, W_lat)
            # Preserve 5D shape for tiling compatibility
            return VAETiler.encode_image_tiled(
                image=image,
                encode_fn=vae.encode,
                latent_channels=getattr(vae, "latent_channels", 16),
                tile_size=(tiling_config.vae_encode_tile_size, tiling_config.vae_encode_tile_size),
                tile_overlap=(tiling_config.vae_encode_tile_overlap, tiling_config.vae_encode_tile_overlap),
                spatial_scale=getattr(vae, "spatial_scale", 8),
            )

        # 2. Standard encoding (fallback)
        encoded = vae.encode(image)

        # 3. Handle dimension fixups (5D -> 4D if needed)
        # Most of our latent processing utilities expect (B, C, H, W) for non-tiled encoding
        if encoded.ndim == 5 and encoded.shape[2] == 1:
            encoded = encoded[:, :, 0, :, :]

        return encoded

    @staticmethod
    def decode(
        vae: nn.Module,
        latent: mx.array,
        tiling_config: TilingConfig | None = None,
    ) -> mx.array:
        # 1. Tiled decoding if enabled
        if (
            tiling_config is not None
            and tiling_config.vae_decode_tiles_per_dim
            and tiling_config.vae_decode_tiles_per_dim > 1
        ):
            # VAETiler expects 5D (B, C, T, H, W)
            if latent.ndim == 4:
                latent = latent[:, :, None, :, :]

            spatial_scale = getattr(vae, "spatial_scale", 8)
            overlap_px = int(tiling_config.vae_decode_overlap) * spatial_scale
            tile_size = int(tiling_config.vae_decode_tile_size)
            return VAETiler.decode_image_tiled(
                latent=latent,
                decode_fn=vae.decode,
                tile_size=(tile_size, tile_size),
                tile_overlap=(overlap_px, overlap_px),
                spatial_scale=spatial_scale,
            )

        # 2. Standard decoding (fallback)
        decoded = vae.decode(latent)

        # 3. Handle dimension fixups (5D -> 4D if needed)
        # Most of our image saving/processing utilities expect (B, C, H, W)
        if decoded.ndim == 5 and decoded.shape[2] == 1:
            decoded = decoded[:, :, 0, :, :]

        return decoded
