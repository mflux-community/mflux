import mlx.core as mx
from mlx import nn


class Attention3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=32, dims=channels, eps=1e-6, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T, H, W = x.shape
        dtype = x.dtype
        residual = x
        x = x.transpose(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H * W)
        x = x.transpose(0, 2, 1)

        x = self.group_norm(x.astype(mx.float32)).astype(dtype)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = q[:, None, :, :]
        k = k[:, None, :, :]
        v = v[:, None, :, :]

        x = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=C**-0.5,
        )

        x = x.squeeze(1)
        x = self.to_out[0](x)
        x = x.transpose(0, 2, 1)
        x = x.reshape(B, T, C, H, W)
        x = x.transpose(0, 2, 1, 3, 4)
        return x + residual
