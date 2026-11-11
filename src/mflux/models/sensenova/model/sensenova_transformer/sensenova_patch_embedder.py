import mlx.core as mx
from mlx import nn


class SenseNovaPatchEmbedder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 1024,
        hidden_size: int = 4096,
        merge_size: int = 2,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rope_theta = rope_theta
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size, bias=True)
        self.dense_embedding = nn.Conv2d(embed_dim, hidden_size, merge_size, stride=merge_size, bias=True)

    def __call__(self, image: mx.array) -> mx.array:
        # image: [B, 3, H, W] -> [B, (H/32)*(W/32), hidden_size], row-major over the token grid
        dtype = self.patch_embedding.weight.dtype
        x = nn.gelu(self.patch_embedding(image.transpose(0, 2, 3, 1).astype(dtype)))
        x = self._rope_2d(x).astype(dtype)
        x = self.dense_embedding(x)
        return x.reshape(x.shape[0], -1, x.shape[-1])

    def _rope_2d(self, x: mx.array) -> mx.array:
        # Interleaved pairs; the first half of the channels rotates by the patch column, the second by the row
        batch, grid_h, grid_w, _ = x.shape
        half = self.embed_dim // 2
        freqs = 1.0 / self.rope_theta ** (mx.arange(0, half, 2, dtype=mx.float32) / half)
        ang_x = mx.arange(grid_w, dtype=mx.float32)[:, None] * freqs[None, :]
        ang_y = mx.arange(grid_h, dtype=mx.float32)[:, None] * freqs[None, :]
        ang_x = mx.broadcast_to(ang_x[None, :, :], (grid_h, grid_w, half // 2))
        ang_y = mx.broadcast_to(ang_y[:, None, :], (grid_h, grid_w, half // 2))
        ang = mx.stack([ang_x, ang_y], axis=2)[None]
        cos, sin = mx.cos(ang), mx.sin(ang)
        pairs = x.astype(mx.float32).reshape(batch, grid_h, grid_w, 2, half // 2, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        rotated = mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1)
        return rotated.reshape(batch, grid_h, grid_w, self.embed_dim)
