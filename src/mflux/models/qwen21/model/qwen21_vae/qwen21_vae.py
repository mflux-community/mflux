import mlx.core as mx

from mflux.models.qwen21.model.qwen21_vae.blocks import VAEOperations
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae_config import Qwen21VAEConfig
from mflux.models.qwen21.model.qwen21_vae.vae import Qwen21VAECore


class Qwen21VAE(Qwen21VAECore):
    LATENTS_MEAN = Qwen21VAEConfig.LATENTS_MEAN
    LATENTS_STD = Qwen21VAEConfig.LATENTS_STD

    def __init__(self, config: dict | None = None):
        super().__init__(config, text_mode=True)

    def decode(self, latents: mx.array) -> mx.array:
        latents = self._single_frame(latents)
        mean = mx.array(self._mean, dtype=mx.float32).reshape(1, self.latent_channels, 1, 1)
        std = mx.array(self._std, dtype=mx.float32).reshape(1, self.latent_channels, 1, 1)
        latents = latents * std + mean
        latents = VAEOperations.conv(self.post_quant_conv, latents, True)
        return self.decoder(latents)[:, :3]

    def encode(self, latents: mx.array) -> mx.array:
        latents = self._single_frame(latents)
        if latents.shape[1] == 3:
            latents = mx.concatenate([latents, mx.ones_like(latents[:, :1])], axis=1)
        latents = VAEOperations.conv(self.quant_conv, self.encoder(latents), True)[:, : self.latent_channels]
        mean = mx.array(self._mean, dtype=mx.float32).reshape(1, self.latent_channels, 1, 1)
        std = mx.array(self._std, dtype=mx.float32).reshape(1, self.latent_channels, 1, 1)
        return (latents - mean) / std
