import mlx.core as mx
from mlx import nn


class Qwen21CausalConv(nn.Module):
    # The 2.1 VAE applies its "3D" convolutions per frame: for single-frame inputs every
    # convolution is a plain 2D conv with symmetric spatial padding (temporal causality
    # only matters for chunked video, which mflux does not run).

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # (B, C, H, W) -> (B, H, W, C) for mlx conv and back
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.conv(x)
        return mx.transpose(x, (0, 3, 1, 2))
