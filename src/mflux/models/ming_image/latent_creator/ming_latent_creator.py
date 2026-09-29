import mlx.core as mx

from mflux.models.common.config import ModelConfig


class MingLatentCreator:
    # Ming keeps latents as plain (1, 16, H/8, W/8) tensors end to end; nothing is packed.
    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return mx.random.normal(shape=[1, 16, height // 8, width // 8], key=mx.random.key(seed)).astype(mx.float32)

    @staticmethod
    def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        return latents

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:  # noqa: ARG004
        return latents.astype(ModelConfig.precision)
