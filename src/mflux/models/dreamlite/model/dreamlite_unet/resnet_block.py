import mlx.core as mx
from mlx import nn

from mflux.models.dreamlite.model.dreamlite_unet.separable_conv import DreamLiteSeparableConv


class DreamLiteResnetBlock(nn.Module):
    # The inverted bottleneck of the pruned UNet: both 3x3 convolutions are depthwise separable, twice as wide inside
    EXPAND_RATIO = 2
    GROUPS = 32
    EPS = 1e-5

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        inner_channels = out_channels * DreamLiteResnetBlock.EXPAND_RATIO
        groups, eps = DreamLiteResnetBlock.GROUPS, DreamLiteResnetBlock.EPS
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, pytorch_compatible=True)
        self.conv1 = DreamLiteSeparableConv(in_channels, inner_channels)
        self.time_emb_proj = nn.Linear(time_embed_dim, inner_channels)
        self.norm2 = nn.GroupNorm(groups, inner_channels, eps=eps, pytorch_compatible=True)
        self.conv2 = DreamLiteSeparableConv(inner_channels, out_channels)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, hidden: mx.array, time_embedding: mx.array) -> mx.array:
        residual = hidden
        hidden = self.conv1(nn.silu(self.norm1(hidden)))
        hidden = hidden + self.time_emb_proj(nn.silu(time_embedding))[:, None, None, :]
        hidden = self.conv2(nn.silu(self.norm2(hidden)))
        if "conv_shortcut" in self:
            residual = self.conv_shortcut(residual)
        return residual + hidden
