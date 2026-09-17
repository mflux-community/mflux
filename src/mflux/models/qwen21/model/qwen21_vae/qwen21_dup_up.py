import mlx.core as mx
from mlx import nn


class Qwen21DupUp(nn.Module):
    # Parameterless shortcut of the residual up blocks: channel repeat + rearrange into
    # nearest (temporal, spatial) upsampling. The duplicated first temporal slot is always
    # dropped so a single-frame input stays single-frame.

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_channels * self.factor % in_channels != 0:
            raise ValueError(
                f"out_channels ({out_channels}) * factor ({self.factor}) must be divisible by in_channels ({in_channels})"
            )
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array) -> mx.array:
        b, _, h, w = x.shape
        ft, fsh, fsw = self.factor_t, self.factor_s, self.factor_s
        x = mx.repeat(x, self.repeats, axis=1)
        x = x.reshape(b, self.out_channels, ft, fsh, fsw, 1, h, w)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))  # (B, out, T, ft, H, fsh, W, fsw)
        x = x.reshape(b, self.out_channels, ft, h * fsh, w * fsw)
        x = x[:, :, ft - 1 :, :, :]
        return mx.squeeze(x, 2)
