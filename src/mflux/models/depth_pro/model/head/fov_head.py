import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.depth_pro_util import DepthProUtil


class FOVHead(nn.Module):
    def __init__(self, dim_decoder: int = 256):
        super().__init__()
        self.convs = [
            nn.Conv2d(in_channels=dim_decoder, out_channels=dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),  # fmt: off
            nn.Conv2d(in_channels=dim_decoder // 2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Identity(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = DepthProUtil.apply_conv(x, self.convs[0])
        x = DepthProUtil.apply_conv(x, self.convs[1])
        x = DepthProUtil.apply_conv(x, self.convs[2])
        x = nn.relu(x)
        x = DepthProUtil.apply_conv(x, self.convs[4])
        x = nn.relu(x)
        return x
