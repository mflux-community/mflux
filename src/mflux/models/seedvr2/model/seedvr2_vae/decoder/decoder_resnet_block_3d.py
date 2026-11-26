import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, dims=in_channels, eps=1e-6, pytorch_compatible=True)
        self.norm2 = nn.GroupNorm(num_groups=32, dims=out_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        residual = x

        x = x.transpose(0, 2, 3, 4, 1)
        x = self.norm1(x.astype(mx.float32)).astype(dtype)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv1(x)

        x = x.transpose(0, 2, 3, 4, 1)
        x = self.norm2(x.astype(mx.float32)).astype(dtype)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return x + residual
