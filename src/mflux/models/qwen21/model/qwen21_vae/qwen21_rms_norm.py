import mlx.core as mx
from mlx import nn


class Qwen21RMSNorm(nn.Module):
    # Wan-style norm: L2-normalize over the channel axis, then scale by sqrt(C) * gamma.
    # Operates on (B, C, H, W); the checkpoint stores gamma with shape (C, 1, 1[, 1]).

    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.scale = float(num_channels) ** 0.5

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        l2_norm = mx.sqrt(mx.sum(x_float * x_float, axis=1, keepdims=True))
        x_normalized = x_float / mx.maximum(l2_norm, 1e-12)
        return (x_normalized * self.scale * self.weight[None, :, None, None]).astype(x.dtype)
