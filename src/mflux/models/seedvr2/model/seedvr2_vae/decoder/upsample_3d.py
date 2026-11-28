import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d


class Upsample3D(nn.Module):
    def __init__(
        self,
        channels: int,
        temporal_up: bool = False,
    ):
        super().__init__()
        spatial_factor = 2
        temporal_factor = 2 if temporal_up else 1
        total_factor = (spatial_factor**2) * temporal_factor

        self.conv = CausalConv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_padding_causal=True,
        )
        self.upscale_conv = CausalConv3d(
            channels,
            channels * total_factor,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T, H, W = x.shape
        sf = self.spatial_factor
        tf = self.temporal_factor
        if T == 1 and tf > 1:
            # Single-frame upsampling discards all temporal slices except t=0,
            # so compute only the kept channels of the 1x1 upscale projection.
            w0 = self.upscale_conv.weight.reshape(sf, sf, tf, C, C)[:, :, 0, :, :].reshape(sf * sf * C, C)
            b0 = self.upscale_conv.bias.reshape(sf, sf, tf, C)[:, :, 0, :].reshape(sf * sf * C)
            x2d = x[:, :, 0, :, :].transpose(0, 2, 3, 1)
            y = mx.addmm(b0, x2d, w0.T)
            y = y.reshape(B, H, W, sf, sf, C).transpose(0, 5, 1, 3, 2, 4).reshape(B, C, 1, H * sf, W * sf)
            return self.conv(y)
        x = self.upscale_conv(x)
        x = x.reshape(B, sf, sf, tf, C, T, H, W)
        x = x.transpose(0, 4, 5, 3, 6, 1, 7, 2)
        x = x.reshape(B, C, T * tf, H * sf, W * sf)
        x = self.conv(x)
        return x
