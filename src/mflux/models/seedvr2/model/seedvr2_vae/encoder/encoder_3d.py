import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d
from mflux.models.seedvr2.model.seedvr2_vae.encoder.down_block_3d import DownBlock3D
from mflux.models.seedvr2.model.seedvr2_vae.encoder.mid_block_3d import MidBlock3D


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        block_out_channels: tuple = (128, 256, 512, 512),
        layers_per_block: int = 2,
        temporal_down_blocks: int = 2,
    ):
        super().__init__()
        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = []
        output_channel = block_out_channels[0]
        num_blocks = len(block_out_channels)

        for i, channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = channel
            is_final_block = i == num_blocks - 1
            temporal_down = (i >= num_blocks - temporal_down_blocks - 1) and not is_final_block

            self.down_blocks.append(
                DownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    add_downsample=not is_final_block,
                    temporal_down=temporal_down,
                )
            )

        self.mid_block = MidBlock3D(channels=block_out_channels[-1])

        self.conv_norm_out = nn.GroupNorm(
            num_groups=32,
            dims=block_out_channels[-1],
            eps=1e-6,
            pytorch_compatible=True,
        )
        self.conv_out = CausalConv3d(
            in_channels=block_out_channels[-1],
            out_channels=2 * out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.mid_block(x)
        dtype = x.dtype
        x = x.transpose(0, 2, 3, 4, 1)
        x = self.conv_norm_out(x.astype(mx.float32)).astype(dtype)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x
