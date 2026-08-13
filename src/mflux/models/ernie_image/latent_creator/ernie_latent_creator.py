import mlx.core as mx

from mflux.models.common.config import ModelConfig


class ErnieLatentCreator:
    LATENT_CHANNELS = 128
    SPATIAL_SCALE = 16

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        h = height // ErnieLatentCreator.SPATIAL_SCALE
        w = width // ErnieLatentCreator.SPATIAL_SCALE
        return mx.random.normal(
            shape=[1, ErnieLatentCreator.LATENT_CHANNELS, h, w],
            key=mx.random.key(seed),
        ).astype(ModelConfig.precision)

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        B, C, H, W = latents.shape
        h, w = H // 2, W // 2
        latents = latents.reshape(B, C, h, 2, w, 2)
        latents = latents.transpose(0, 1, 3, 5, 2, 4)
        return latents.reshape(B, C * 4, h, w)

    @staticmethod
    def bn_normalize_latents(latents: mx.array, vae) -> mx.array:
        bn_mean = vae.bn.running_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(vae.bn.running_var.reshape(1, -1, 1, 1) + vae.bn.eps)
        return (latents - bn_mean) / bn_std

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        return latents
