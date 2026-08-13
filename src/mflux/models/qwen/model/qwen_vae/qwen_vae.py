import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_decoder_3d import QwenImageDecoder3D
from mflux.models.qwen.model.qwen_vae.qwen_image_encoder_3d import QwenImageEncoder3D


class QwenVAE(nn.Module):
    LATENTS_MEAN = np.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921], dtype=np.float32)  # fmt: off
    LATENTS_STD = np.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916], dtype=np.float32)  # fmt: off
    spatial_scale = 8
    latent_channels = 16
    pid_variant = "qwen-image"  # see PID_CHECKPOINT_VARIANTS

    def __init__(self):
        super().__init__()
        self.decoder = QwenImageDecoder3D()
        self.encoder = QwenImageEncoder3D()
        self.post_quant_conv = QwenImageCausalConv3D(16, 16, 1, 1, 0)
        self.quant_conv = QwenImageCausalConv3D(32, 32, 1, 1, 0)  # Keep 32 channels like diffusers

    def decode(self, latents: mx.array) -> mx.array:
        if len(latents.shape) == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        latents_mean = mx.array(self.LATENTS_MEAN).reshape(1, self.latent_channels, 1, 1, 1)
        latents_std = mx.array(self.LATENTS_STD).reshape(1, self.latent_channels, 1, 1, 1)
        latents = latents * latents_std + latents_mean
        latents = self.post_quant_conv(latents)
        decoded = self.decoder(latents)
        return decoded

    def encode(self, latents: mx.array) -> mx.array:
        if len(latents.shape) == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        latents = self.encoder(latents)
        latents = self.quant_conv(latents)
        latents = latents[:, :16, :, :, :]
        latents_mean = mx.array(self.LATENTS_MEAN).reshape(1, self.latent_channels, 1, 1, 1)
        latents_std = mx.array(self.LATENTS_STD).reshape(1, self.latent_channels, 1, 1, 1)
        return (latents - latents_mean) / latents_std
