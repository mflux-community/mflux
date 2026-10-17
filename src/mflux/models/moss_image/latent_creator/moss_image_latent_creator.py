import mlx.core as mx


class MOSSImageLatentCreator:
    @staticmethod
    def create_latents(batch_size: int, height: int, width: int, seed: int) -> mx.array:
        key = mx.random.key(seed)
        shape = (batch_size, 128, height // 16, width // 16)
        return mx.random.normal(shape=shape, key=key)

    @staticmethod
    def unpack_for_vae(latents: mx.array, vae_bn_mean=None, vae_bn_var=None, eps: float = 1e-4) -> mx.array:
        # latents: (B, 128, H, W)
        # 1. BN un-normalization if provided
        z = latents
        if vae_bn_mean is not None and vae_bn_var is not None:
            scale = mx.sqrt(vae_bn_var + eps)
            z = z * mx.reshape(scale, (1, -1, 1, 1)) + mx.reshape(vae_bn_mean, (1, -1, 1, 1))

        # 2. unpatch: (B, 128, H, W) -> (B, 32, 2, 2, H, W) -> permute -> (B, 32, 2*H, 2*W)
        B, C, H, W = z.shape
        z = mx.reshape(z, (B, C // 4, 2, 2, H, W))
        # permute(0, 1, 4, 2, 5, 3) in PyTorch (B, c//4, H, 2, W, 2)
        z = mx.transpose(z, (0, 1, 4, 2, 5, 3))
        z = mx.reshape(z, (B, C // 4, 2 * H, 2 * W))
        return z
