import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_vae.qwen21_causal_conv import Qwen21CausalConv
from mflux.models.qwen21.model.qwen21_vae.qwen21_rms_norm import Qwen21RMSNorm


class Qwen21ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = Qwen21RMSNorm(in_dim)
        self.conv1 = Qwen21CausalConv(in_dim, out_dim, 3, 1)
        self.norm2 = Qwen21RMSNorm(out_dim)
        self.conv2 = Qwen21CausalConv(out_dim, out_dim, 3, 1)
        self.conv_shortcut = Qwen21CausalConv(in_dim, out_dim, 1, 0) if in_dim != out_dim else None

    def __call__(self, x: mx.array) -> mx.array:
        residual = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = nn.silu(self.norm1(x))
        x = self.conv1(x)
        x = nn.silu(self.norm2(x))
        x = self.conv2(x)
        return x + residual
