import mlx.core as mx
from PIL import Image

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.image_util import ImageUtil


class Fibo2LatentCreator:
    CHANNELS = 32
    VAE_SCALE_FACTOR = 8

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        shape = (1, Fibo2LatentCreator.CHANNELS, height // Fibo2LatentCreator.VAE_SCALE_FACTOR, width // Fibo2LatentCreator.VAE_SCALE_FACTOR)  # fmt: off
        return mx.random.normal(shape=shape, key=mx.random.key(seed), dtype=mx.float32)

    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        return latents

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        # The packed (2x2-patchified) latents, which callbacks decode with the VAE's BatchNorm statistics
        return Flux2LatentCreator.patchify_latents(latents)

    @staticmethod
    def decode(vae: Flux2VAE, latents: mx.array, tiling_config: TilingConfig | None = None) -> mx.array:
        # The latents are normalized with the VAE's BatchNorm statistics in the 2x2-patchified space
        packed = Flux2LatentCreator.patchify_latents(latents)
        return vae.decode_packed_latents(packed, tiling_config=tiling_config)

    @staticmethod
    def encode_image(vae: Flux2VAE, image: Image.Image, tiling_config: TilingConfig | None = None) -> mx.array:
        # The image to edit, normalized with the VAE's BatchNorm statistics like the latents being generated
        pixels = ImageUtil.to_array(image).astype(ModelConfig.precision)
        latents = Fibo2LatentCreator.pack_latents(VAEUtil.encode(vae, pixels, tiling_config=tiling_config))
        return vae.normalize_latents(latents).astype(ModelConfig.precision)
