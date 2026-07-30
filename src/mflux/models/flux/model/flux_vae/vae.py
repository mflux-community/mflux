import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_vae.decoder.decoder import Decoder
from mflux.models.flux.model.flux_vae.encoder.encoder import Encoder


class VAE(nn.Module):
    scaling_factor: int = 0.3611
    shift_factor: int = 0.1159
    spatial_scale = 8
    latent_channels = 16
    pid_variant = "flux"  # see PID_CHECKPOINT_VARIANTS

    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        scaled_latents = (latents / self.scaling_factor) + self.shift_factor
        decoded = self.decoder(scaled_latents)
        return decoded[:, :, None, :, :]

    def encode(self, image: mx.array) -> mx.array:
        if image.ndim == 5:
            image = image[:, :, 0, :, :]
        latents = self.encoder(image)
        mean, _ = mx.split(latents, 2, axis=1)
        latent = (mean - self.shift_factor) * self.scaling_factor
        return latent[:, :, None, :, :]
