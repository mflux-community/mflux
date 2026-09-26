import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.qwen21.model.qwen21_vae.qwen21_causal_conv import Qwen21CausalConv
from mflux.models.qwen21.model.qwen21_vae.qwen21_decoder import Qwen21Decoder
from mflux.models.qwen21.model.qwen21_vae.qwen21_encoder import Qwen21Encoder


class Qwen21VAE(nn.Module):
    # fmt: off
    LATENTS_MEAN = np.array([
        0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
        1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
        2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
        -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
        -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.107, 1.7376,
        -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
        -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
        0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1566, 0.0304,
    ], dtype=np.float32)  # fmt: on
    # fmt: off
    LATENTS_STD = np.array([
        3.2001, 3.2936, 3.4321, 3.0091, 3.106, 4.0379, 4.0705, 3.791,
        3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756,
        3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
        3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467,
        3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
        3.532, 4.7833, 3.1284, 4.181, 3.8527, 3.8317, 3.5603, 4.3867,
        3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.44, 3.4957,
        3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
    ], dtype=np.float32)  # fmt: on
    spatial_scale = 16
    latent_channels = 64

    def __init__(self):
        super().__init__()
        self.encoder = Qwen21Encoder()
        self.quant_conv = Qwen21CausalConv(128, 128, 1, 0)
        self.post_quant_conv = Qwen21CausalConv(64, 64, 1, 0)
        self.decoder = Qwen21Decoder()

    def decode(self, latents: mx.array, keep_alpha: bool = False) -> mx.array:
        if latents.shape[-3] == 1 and latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        latents_mean = mx.array(self.LATENTS_MEAN).reshape(1, self.latent_channels, 1, 1)
        latents_std = mx.array(self.LATENTS_STD).reshape(1, self.latent_channels, 1, 1)
        latents = latents * latents_std + latents_mean
        latents = self.post_quant_conv(latents)
        decoded = self.decoder(latents)
        # The 2.1 VAE has 4 output channels (RGBA). The alpha carries edit masks /
        # transparency depending on the checkpoint's training; the default drops it
        # (RGB output), keep_alpha=True returns the decoder's RGBA stream.
        if keep_alpha or decoded.shape[1] < 4:
            return decoded
        return decoded[:, :3, :, :]

    def encode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        if latents.shape[1] == 3:
            alpha = mx.ones_like(latents[:, :1, :, :])
            latents = mx.concatenate([latents, alpha], axis=1)
        latents = self.encoder(latents)
        latents = self.quant_conv(latents)
        latents = latents[:, : self.latent_channels, :, :]
        latents_mean = mx.array(self.LATENTS_MEAN).reshape(1, self.latent_channels, 1, 1)
        latents_std = mx.array(self.LATENTS_STD).reshape(1, self.latent_channels, 1, 1)
        return (latents - latents_mean) / latents_std
