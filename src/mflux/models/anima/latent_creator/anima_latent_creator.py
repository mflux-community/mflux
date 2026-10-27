import mlx.core as mx


class AnimaLatentCreator:
    # Qwen-Image VAE latents: 16 channels at 1/8 of the image size. The transformer takes them as they are and
    # patches them itself, so packing is only the 5D VAE output squeezed to 4D.
    LATENT_CHANNELS = 16
    SIZE_MULTIPLE = 16

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return mx.random.normal(
            shape=(1, AnimaLatentCreator.LATENT_CHANNELS, height // 8, width // 8),
            key=mx.random.key(seed),
            dtype=mx.float32,
        )

    @staticmethod
    def check_size(height: int, width: int) -> None:
        multiple = AnimaLatentCreator.SIZE_MULTIPLE
        if height <= 0 or width <= 0 or height % multiple or width % multiple:
            raise ValueError(f"Anima needs a width and height that are positive multiples of {multiple}, got {width}x{height}.")  # fmt: off

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        return latents

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        return latents
