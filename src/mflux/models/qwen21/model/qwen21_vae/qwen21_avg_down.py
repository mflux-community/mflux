import mlx.core as mx
from mlx import nn


class Qwen21AvgDown(nn.Module):
    # Parameterless shortcut of the residual down blocks: average pooling over the
    # (temporal, spatial) factor, then a channel-group mean that maps in -> out channels.
    # Single-frame inputs with factor_t=2 are front-padded with a zero frame (causal pad),
    # exactly like the reference.

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels != 0:
            raise ValueError(
                f"in_channels ({in_channels}) * factor ({self.factor}) must be divisible by out_channels ({out_channels})"
            )
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        b, c, h, w = x.shape
        hs, ws = self.factor_s, self.factor_s
        x = mx.expand_dims(x, 2)  # (B, C, 1, H, W)
        if self.factor_t == 2:
            x = mx.pad(x, [(0, 0), (0, 0), (1, 0), (0, 0), (0, 0)])
        x = x.reshape(b, c, 1, self.factor_t, h // hs, hs, w // ws, ws)
        x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))  # (B, C, ft, fsh, fsw, T', H', W')
        x = x.reshape(b, c * self.factor, h // hs, w // ws)
        x = x.reshape(b, self.out_channels, self.group_size, h // hs, w // ws)
        return mx.mean(x, axis=2)
