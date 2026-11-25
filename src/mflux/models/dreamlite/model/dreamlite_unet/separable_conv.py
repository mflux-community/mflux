import mlx.core as mx
from mlx import nn


class DreamLiteSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.pointwise(self.depthwise(hidden))
