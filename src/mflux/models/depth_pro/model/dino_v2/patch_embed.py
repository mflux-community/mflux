import mlx.core as mx
import mlx.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 1024, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)  # fmt: off

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.proj(x)
        return x
