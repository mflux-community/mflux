import mlx.core as mx
from mlx import nn

from mflux.models.dreamlite.model.dreamlite_vae.tiny_block import DreamLiteTinyBlock


class DreamLiteTinyEncoder(nn.Module):
    NUM_BLOCKS = (1, 3, 3, 3)

    def __init__(self, channels: int, latent_channels: int):
        super().__init__()
        layers: list[nn.Module] = []
        for index, num_blocks in enumerate(DreamLiteTinyEncoder.NUM_BLOCKS):
            if index == 0:
                layers.append(nn.Conv2d(3, channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2, bias=False))
            layers.extend(DreamLiteTinyBlock(channels) for _ in range(num_blocks))
        layers.append(nn.Conv2d(channels, latent_channels, kernel_size=3, padding=1))
        self.layers = layers

    def __call__(self, image: mx.array) -> mx.array:
        hidden = (image + 1) / 2
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class DreamLiteTinyDecoder(nn.Module):
    NUM_BLOCKS = (3, 3, 3, 1)

    def __init__(self, channels: int, latent_channels: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(latent_channels, channels, kernel_size=3, padding=1), nn.ReLU()]
        last = len(DreamLiteTinyDecoder.NUM_BLOCKS) - 1
        for index, num_blocks in enumerate(DreamLiteTinyDecoder.NUM_BLOCKS):
            layers.extend(DreamLiteTinyBlock(channels) for _ in range(num_blocks))
            if index < last:
                layers.append(nn.Upsample(scale_factor=2.0, mode="nearest"))
                layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
            else:
                layers.append(nn.Conv2d(channels, 3, kernel_size=3, padding=1))
        self.layers = layers

    def __call__(self, latents: mx.array) -> mx.array:
        hidden = mx.tanh(latents / 3) * 3
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden * 2 - 1


class DreamLiteVAE(nn.Module):
    # TAESDXL (diffusers' AutoencoderTiny): 2.4M weights, 8x per side, latents used unscaled
    CHANNELS = 64
    latent_channels = 4
    spatial_scale = 8
    # A full 1024x1024 decode peaks near 1 GB, so memory-saving modes have nothing to gain from tiles
    supports_implicit_tiling = False

    def __init__(self):
        super().__init__()
        self.encoder = DreamLiteTinyEncoder(DreamLiteVAE.CHANNELS, DreamLiteVAE.latent_channels)
        self.decoder = DreamLiteTinyDecoder(DreamLiteVAE.CHANNELS, DreamLiteVAE.latent_channels)

    def encode(self, image: mx.array) -> mx.array:
        dtype = self.encoder.layers[0].weight.dtype
        return self.encoder(image.astype(dtype).transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

    def decode(self, latents: mx.array) -> mx.array:
        dtype = self.decoder.layers[0].weight.dtype
        return self.decoder(latents.astype(dtype).transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
