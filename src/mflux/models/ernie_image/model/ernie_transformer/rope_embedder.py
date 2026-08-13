import mlx.core as mx
from mlx import nn


def _rope(pos: mx.array, dim: int, theta: int) -> mx.array:
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta**scale)
    return pos[..., None].astype(mx.float32) * omega


class ErnieRopeEmbedder(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> mx.array:
        parts = [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)]
        emb = mx.concatenate(parts, axis=-1)
        emb = emb[:, :, None, :]
        emb = mx.stack([emb, emb], axis=-1)
        return emb.reshape(*emb.shape[:-2], -1)


def apply_rotary_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rotated = mx.concatenate([-q2, q1], axis=-1)
    k_rotated = mx.concatenate([-k2, k1], axis=-1)
    return q * cos + q_rotated * sin, k * cos + k_rotated * sin
