import mlx.core as mx
from PIL import Image

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.dreamlite.model.dreamlite_vae.vae import DreamLiteVAE
from mflux.utils.image_util import ImageUtil


class DreamLiteLatentCreator:
    CHANNELS = 4
    VAE_SCALE_FACTOR = 8
    # The UNet halves the latents twice, so each side is a multiple of 8 * 4 pixels
    SIZE_MULTIPLE = 32

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        scale = DreamLiteLatentCreator.VAE_SCALE_FACTOR
        shape = (1, DreamLiteLatentCreator.CHANNELS, height // scale, width // scale)
        return mx.random.normal(shape=shape, key=mx.random.key(seed), dtype=mx.float32)

    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        return latents

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents

    @staticmethod
    def encode_image(vae: DreamLiteVAE, image: Image.Image, height: int, width: int) -> mx.array:
        # TAESDXL latents are used as they come: no scaling and no shift. It encodes the whole image, never tiles
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        encoded = VAEUtil.encode(vae, ImageUtil.to_array(resized))
        return DreamLiteLatentCreator.pack_latents(encoded).astype(mx.float32)

    @staticmethod
    def with_context(latents: mx.array, context_latents: mx.array) -> mx.array:
        # In-context spatial concatenation: the image being generated on the left, the context image on the right
        return mx.concatenate([latents, context_latents], axis=3)

    @staticmethod
    def check_size(height: int, width: int) -> None:
        multiple = DreamLiteLatentCreator.SIZE_MULTIPLE
        if height <= 0 or width <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        if height % multiple or width % multiple:
            raise ValueError(f"width and height must be multiples of {multiple}, got {width}x{height}")
