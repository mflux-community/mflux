import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d
from mflux.models.seedvr2.model.seedvr2_vae.decoder.decoder_mid_block_3d import MidBlock3D
from mflux.models.seedvr2.model.seedvr2_vae.decoder.up_block_3d import UpBlock3D


class Decoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: tuple = (128, 256, 512, 512),
        layers_per_block: int = 3,
        temporal_up_blocks: int = 2,
    ):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))

        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=reversed_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = MidBlock3D(
            channels=reversed_channels[0],
        )

        self.up_blocks = []
        output_channel = reversed_channels[0]
        num_blocks = len(reversed_channels)

        for i, channel in enumerate(reversed_channels):
            input_channel = output_channel
            output_channel = channel
            is_final_block = i == num_blocks - 1
            temporal_up = i < temporal_up_blocks

            self.up_blocks.append(
                UpBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    add_upsample=not is_final_block,
                    temporal_up=temporal_up,
                )
            )

        self.conv_norm_out = nn.GroupNorm(
            num_groups=32,
            dims=reversed_channels[-1],
            eps=1e-6,
            pytorch_compatible=True,
        )
        self.conv_out = CausalConv3d(
            in_channels=reversed_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        dtype = x.dtype
        x = x.transpose(0, 2, 3, 4, 1)
        x = self.conv_norm_out(x.astype(mx.float32)).astype(dtype)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x
