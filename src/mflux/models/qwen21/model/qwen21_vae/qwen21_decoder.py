import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_vae.qwen21_causal_conv import Qwen21CausalConv
from mflux.models.qwen21.model.qwen21_vae.qwen21_dup_up import Qwen21DupUp
from mflux.models.qwen21.model.qwen21_vae.qwen21_mid_block import Qwen21MidBlock
from mflux.models.qwen21.model.qwen21_vae.qwen21_res_block import Qwen21ResBlock
from mflux.models.qwen21.model.qwen21_vae.qwen21_resample import Qwen21Resample
from mflux.models.qwen21.model.qwen21_vae.qwen21_rms_norm import Qwen21RMSNorm


class Qwen21ResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_upsample: bool,
        up_flag: bool = True,
    ):
        super().__init__()
        self.resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(Qwen21ResBlock(current_dim, out_dim))
            current_dim = out_dim
        self.upsampler = Qwen21Resample(out_dim, out_dim, "upsample") if up_flag else None
        self.avg_shortcut = Qwen21DupUp(in_dim, out_dim, factor_t=2 if temporal_upsample else 1) if up_flag else None

    def __call__(self, x: mx.array) -> mx.array:
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
            return x + self.avg_shortcut(x_copy)
        return x


class Qwen21Decoder(nn.Module):
    def __init__(
        self,
        dim: int = 144,
        z_dim: int = 64,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temporal_upsample: tuple[bool, ...] = (True, True, True, False),
        out_channels: int = 4,
    ):
        super().__init__()
        dims = [dim * mult for mult in [dim_mult[-1]] + list(dim_mult[::-1])]

        self.conv_in = Qwen21CausalConv(z_dim, dims[0], 3, 1)
        self.mid_block = Qwen21MidBlock(dims[0])
        # dims has one more entry than dim_mult (the doubled top multiplier), giving five
        # transitions; only the first four upsample, the last is a plain residual block group.
        self.up_blocks = [
            Qwen21ResidualUpBlock(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                num_res_blocks=num_res_blocks,
                temporal_upsample=temporal_upsample[i] if i < len(temporal_upsample) else False,
                up_flag=i < len(dims) - 2,
            )
            for i in range(len(dims) - 1)
        ]
        self.norm_out = Qwen21RMSNorm(dims[-1])
        self.conv_out = Qwen21CausalConv(dims[-1], out_channels, 3, 1)

    def __call__(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = nn.silu(self.norm_out(x))
        return self.conv_out(x)
