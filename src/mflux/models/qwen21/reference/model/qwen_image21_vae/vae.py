import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.reference.model.qwen_image21_vae.blocks import ChannelNorm, DownBlock, MidBlock, UpBlock


class Encoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dims = [config["base_dim"] * factor for factor in [1, *config["dim_mult"]]]
        self.conv_in = nn.Conv2d(config["in_channels"], dims[0], 3, padding=1)
        temporal = config["temporal_downsample"]
        self.down_blocks = [
            DownBlock(a, b, config["num_res_blocks"], i < len(dims) - 2, temporal[i] if i < len(temporal) else False)
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.mid_block = MidBlock(dims[-1])
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = nn.Conv2d(dims[-1], config["z_dim"] * 2, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
            mx.eval(x)
        return self.conv_out(nn.silu(self.norm_out(self.mid_block(x))))


class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dim_mult = config["dim_mult"]
        dims = [config["decoder_base_dim"] * factor for factor in [dim_mult[-1], *reversed(dim_mult)]]
        self.conv_in = nn.Conv2d(config["z_dim"], dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])
        temporal = list(reversed(config["temporal_downsample"]))
        self.up_blocks = [
            UpBlock(a, b, config["num_res_blocks"], i < len(dims) - 2, temporal[i] if i < len(temporal) else False)
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = nn.Conv2d(dims[-1], config["out_channels"], 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
            mx.eval(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class QwenImage21VAE(nn.Module):
    spatial_scale = 16
    latent_channels = 64

    def __init__(self, config: dict):
        super().__init__()
        if not config.get("is_residual") or config.get("patch_size") is not None:
            raise ValueError("Qwen-Image-2.1 requires the residual, unpatched image VAE.")
        if "temporal_downsample" not in config:
            # Published Qwen checkpoints and the pinned Diffusers reference use this legacy spelling.
            config = {**config, "temporal_downsample": config["temperal_downsample"]}
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        channels = config["z_dim"]
        self.quant_conv = nn.Conv2d(channels * 2, channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(channels, channels, 1)
        self._mean = tuple(config["latents_mean"])
        self._std = tuple(config["latents_std"])

    def encode(self, image: mx.array) -> mx.array:
        image = self._to_nhwc(image)
        mean = self.quant_conv(self.encoder(image))[..., : len(self._mean)]
        latent = (mean - mx.array(self._mean, dtype=mean.dtype)) / mx.array(self._std, dtype=mean.dtype)
        return latent.transpose(0, 3, 1, 2)[:, :, None]

    def decode(self, latents: mx.array) -> mx.array:
        latents = self._to_nhwc(latents)
        latents = latents * mx.array(self._std, dtype=latents.dtype) + mx.array(self._mean, dtype=latents.dtype)
        image = self.decoder(self.post_quant_conv(latents))
        return mx.clip(image, -1, 1).transpose(0, 3, 1, 2)[:, :, None]

    @staticmethod
    def _to_nhwc(value: mx.array) -> mx.array:
        if value.ndim == 5:
            if value.shape[2] != 1:
                raise ValueError("Qwen-Image-2.1 supports single-frame images only.")
            value = value[:, :, 0]
        return value.transpose(0, 2, 3, 1)
