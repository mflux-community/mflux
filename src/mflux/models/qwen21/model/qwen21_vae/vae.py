import mlx.core as mx
from mlx import nn

from mflux.models.common.vae.block_decoder import VAEBlockDecoder
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

    # Row strips (strips=True). After the mid block, the only global operation of the decoder, each layer is a
    # 3x3 convolution, a nearest upsample or a per-pixel channel norm. So an output row depends only on the rows
    # near it, and the decoder can run on strips of rows with context rows above and below each strip.
    # The decode has two passes. Pass 1 runs the first STRIP_FROM up blocks (low resolution, many channels) on
    # strips of LOW_ROWS latent rows and gives their full output. Pass 2 runs the remaining blocks and the head on
    # strips of that output. The context rows of each pass cover the measured reach of the blocks in that pass,
    # so the pixels agree with the full-image decode up to float rounding. Only one strip of each pass is in
    # memory at a time.
    STRIP_FROM = 2
    LOW_ROWS = 16
    # Context rows for pass 1, by the number of blocks in it. Measured reach of up_blocks[:k] in latent rows:
    # k=1 7.0, k=2 10.5, k=3 12.3.
    LOW_HALO = (0, 8, 12, 14, 16)
    # Context rows for pass 2, by its first block k. Measured reach of up_blocks[k:] and the head, in rows of the
    # input of block k: 13.6, 13.1, 12.3, 10.5, 7.0.
    HALO = (16, 16, 16, 12, 8)
    # Pass 2 strip height: the largest power of two from 16 to 64 with rows * width <= STRIP_BUDGET (width of the
    # input of block STRIP_FROM). This keeps the memory of one strip about constant, and narrow images use fewer
    # strips: 64 rows at 1024 px wide, 32 at 2048 px, 16 at 2400-2752 px.
    STRIP_BUDGET = 16384

    def __call__(self, x: mx.array, strips: bool = False) -> mx.array:
        x = self.mid_block(VAEOperations.conv(self.conv_in, x, self.text_mode))
        if not strips:
            return self._up(x, 0)
        # rows are axis 2 in the NCHW layout (text_mode), axis 1 in NHWC
        axis = 2 if self.text_mode else 1
        first = Decoder.STRIP_FROM
        scale_low = 2 ** sum(block.upsampler is not None for block in self.up_blocks[:first])
        x = Decoder._in_strips(
            x, lambda v: self._blocks(v, 0, first), Decoder.LOW_ROWS, Decoder.LOW_HALO[first], scale_low, axis
        )
        rows = Decoder.strip_rows(x.shape[axis + 1])
        scale_high = 2 ** sum(block.upsampler is not None for block in self.up_blocks[first:])
        return Decoder._in_strips(x, lambda v: self._up(v, first), rows, Decoder.HALO[first], scale_high, axis)

    @staticmethod
    def strip_rows(width: int) -> int:
        rows = 64
        while rows > 16 and rows * width > Decoder.STRIP_BUDGET:
            rows //= 2
        return rows

    @staticmethod
    def _in_strips(x: mx.array, fn, rows: int, halo: int, scale: int, axis: int) -> mx.array:
        mx.eval(x)
        height = x.shape[axis]
        if height <= rows + 2 * halo:
            return fn(x)
        lead = (slice(None),) * axis
        strips = []
        for top in range(0, height, rows):
            bottom = min(height, top + rows)
            a, b = max(0, top - halo), min(height, bottom + halo)
            strip = fn(x[lead + (slice(a, b),)])[lead + (slice((top - a) * scale, (bottom - a) * scale),)]
            mx.eval(strip)
            strips.append(strip)
        return mx.concatenate(strips, axis=axis)

    def _blocks(self, x: mx.array, start: int, stop: int) -> mx.array:
        for block in self.up_blocks[start:stop]:
            x = block(x)
            mx.eval(x)
        return x

    def _up(self, x: mx.array, start: int) -> mx.array:
        for block in self.up_blocks[start:]:
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
    # "low_memory" (the default) decodes in row strips (see Decoder.STRIP_FROM). "full_precision"
    # (--vae-full-precision) and "legacy" decode the full image at once.
    decode_mode: str = VAEBlockDecoder.LOW_MEMORY

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
        image = self.decoder(self.post_quant_conv(latents), strips=self.decode_mode == VAEBlockDecoder.LOW_MEMORY)
        return mx.clip(image, -1, 1).transpose(0, 3, 1, 2)[:, :, None]

    @staticmethod
    def _to_nhwc(value: mx.array) -> mx.array:
        return Qwen21VAECore._single_frame(value).transpose(0, 2, 3, 1)
