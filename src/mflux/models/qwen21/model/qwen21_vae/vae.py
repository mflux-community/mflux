import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_vae.blocks import ChannelNorm, DownBlock, MidBlock, UpBlock, VAEOperations
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae_config import Qwen21VAEConfig


class Encoder(nn.Module):
    def __init__(self, config: dict, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        dims = [config["base_dim"] * factor for factor in [1, *config["dim_mult"]]]
        self.conv_in = nn.Conv2d(config["in_channels"], dims[0], 3, padding=1)
        temporal = config["temporal_downsample"]
        self.down_blocks = [
            DownBlock(
                a,
                b,
                config["num_res_blocks"],
                i < len(dims) - 2,
                temporal[i] if i < len(temporal) else False,
                text_mode,
            )
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.mid_block = MidBlock(dims[-1], text_mode)
        self.norm_out = ChannelNorm(dims[-1], text_mode)
        self.conv_out = nn.Conv2d(dims[-1], config["z_dim"] * 2, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = VAEOperations.conv(self.conv_in, x, self.text_mode)
        for block in self.down_blocks:
            x = block(x)
            if not self.text_mode:
                mx.eval(x)
        return VAEOperations.conv(self.conv_out, nn.silu(self.norm_out(self.mid_block(x))), self.text_mode)


class Decoder(nn.Module):
    def __init__(self, config: dict, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        dim_mult = config["dim_mult"]
        dims = [config["decoder_base_dim"] * factor for factor in [dim_mult[-1], *reversed(dim_mult)]]
        self.conv_in = nn.Conv2d(config["z_dim"], dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0], text_mode)
        temporal = list(reversed(config["temporal_downsample"]))
        self.up_blocks = [
            UpBlock(
                a,
                b,
                config["num_res_blocks"],
                i < len(dims) - 2,
                temporal[i] if i < len(temporal) else False,
                text_mode,
            )
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.norm_out = ChannelNorm(dims[-1], text_mode)
        self.conv_out = nn.Conv2d(dims[-1], config["out_channels"], 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mid_block(VAEOperations.conv(self.conv_in, x, self.text_mode))
        for block in self.up_blocks:
            x = block(x)
            if not self.text_mode:
                mx.eval(x)
        return VAEOperations.conv(self.conv_out, nn.silu(self.norm_out(x)), self.text_mode)


class Qwen21VAECore(nn.Module):
    spatial_scale = 16
    latent_channels = 64

    def __init__(self, config: dict | None = None, *, text_mode: bool = False):
        super().__init__()
        config = Qwen21VAEConfig.resolve(config)
        if not config["is_residual"] or config["patch_size"] is not None:
            raise ValueError("Qwen-Image-2.1 requires the residual, unpatched image VAE.")
        self.encoder = Encoder(config, text_mode)
        self.decoder = Decoder(config, text_mode)
        self.latent_channels = config["z_dim"]
        self.spatial_scale = 2 ** (len(config["dim_mult"]) - 1)
        self.quant_conv = nn.Conv2d(self.latent_channels * 2, self.latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, 1)
        self._mean = tuple(config["latents_mean"])
        self._std = tuple(config["latents_std"])
        if len(self._mean) != self.latent_channels or len(self._std) != self.latent_channels:
            raise ValueError("VAE latent mean and std must match z_dim.")

    @staticmethod
    def _single_frame(value: mx.array) -> mx.array:
        if value.ndim == 5:
            if value.shape[2] != 1:
                raise ValueError("Qwen-Image-2.1 supports single-frame images only.")
            value = value[:, :, 0]
        if value.ndim != 4:
            raise ValueError("Qwen-Image-2.1 expects NCHW or NC1HW images.")
        return value


class QwenImage21VAE(Qwen21VAECore):
    def __init__(self, config: dict):
        if not config.get("is_residual") or config.get("patch_size") is not None:
            raise ValueError("Qwen-Image-2.1 requires the residual, unpatched image VAE.")
        super().__init__(config)

    def encode(self, image: mx.array) -> mx.array:
        image = self._to_nhwc(image)
        mean = self.quant_conv(self.encoder(image))[..., : self.latent_channels]
        latent = (mean - mx.array(self._mean, dtype=mean.dtype)) / mx.array(self._std, dtype=mean.dtype)
        return latent.transpose(0, 3, 1, 2)[:, :, None]

    def decode(self, latents: mx.array) -> mx.array:
        latents = self._to_nhwc(latents)
        latents = latents * mx.array(self._std, dtype=latents.dtype) + mx.array(self._mean, dtype=latents.dtype)
        image = self.decoder(self.post_quant_conv(latents))
        return mx.clip(image, -1, 1).transpose(0, 3, 1, 2)[:, :, None]

    @staticmethod
    def _to_nhwc(value: mx.array) -> mx.array:
        return Qwen21VAECore._single_frame(value).transpose(0, 2, 3, 1)
