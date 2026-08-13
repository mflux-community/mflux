import mlx.core as mx


class Krea2LatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return mx.random.normal(
            shape=(1, 16, height // 8, width // 8),
            key=mx.random.key(seed),
        )

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        return latents

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents
