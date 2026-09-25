import mlx.core as mx

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE


class MingVAE(QwenVAE):
    """Qwen-Image's AutoencoderKLQwenImage retrained for RGBA (input_channels = 4). Latents are
    normalised with a single scaling factor instead of Qwen-Image's per-channel mean/std."""

    SCALING_FACTOR = 8.0064
    image_channels = 4

    def __init__(self):
        super().__init__()
        self.decoder.conv_out = QwenImageCausalConv3D(96, self.image_channels, 3, 1, 1)
        self.encoder.conv_in = QwenImageCausalConv3D(self.image_channels, 96, 3, 1, 1)

    def decode(self, latents: mx.array) -> mx.array:
        if len(latents.shape) == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        latents = self.post_quant_conv(latents / self.SCALING_FACTOR)
        return self.decoder(latents)

    def encode(self, image: mx.array) -> mx.array:
        if len(image.shape) == 4:
            image = image.reshape(image.shape[0], image.shape[1], 1, image.shape[2], image.shape[3])
        latents = self.quant_conv(self.encoder(image))[:, : self.latent_channels]
        return latents * self.SCALING_FACTOR
