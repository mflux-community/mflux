import math

import mlx.core as mx


class SenseNovaLatentCreator:
    CHANNELS = 3
    TOKEN_SIZE = 32
    NOISE_SCALE_MAX = 16.0

    @staticmethod
    def noise_scale(height: int, width: int) -> float:
        tokens = (height // SenseNovaLatentCreator.TOKEN_SIZE) * (width // SenseNovaLatentCreator.TOKEN_SIZE)
        return min(math.sqrt(tokens / 64), SenseNovaLatentCreator.NOISE_SCALE_MAX)

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        shape = (1, SenseNovaLatentCreator.CHANNELS, height, width)
        noise = mx.random.normal(shape=shape, dtype=mx.float32, key=mx.random.key(seed))
        return SenseNovaLatentCreator.noise_scale(height, width) * noise

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents

    @staticmethod
    def check_size(height: int, width: int) -> None:
        size = SenseNovaLatentCreator.TOKEN_SIZE
        if height <= 0 or width <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        if height % size or width % size:
            raise ValueError(f"width and height must be multiples of {size}")
