import mlx.core as mx
from mlx import nn


class Qwen21Resample(nn.Module):
    # Spatial 2x resampling. The "3d" modes of the reference collapse to the same spatial
    # operation on the single-frame path (their time_conv only runs with a feature cache).

    def __init__(self, dim: int, out_dim: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "downsample":
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2)
        elif mode == "upsample":
            self.conv = nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        if self.mode == "upsample":
            x = mx.repeat(x, 2, axis=1)
            x = mx.repeat(x, 2, axis=2)
        else:
            # asymmetric pad (right, bottom) before the stride-2 conv, like ZeroPad2d((0,1,0,1))
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.conv(x)
        return mx.transpose(x, (0, 3, 1, 2))
