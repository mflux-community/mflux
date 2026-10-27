import mlx.core as mx


class AnimaRoPE:
    # The Cosmos-Predict2 3D rotary embedding. The head dimension splits into time, height and width bands,
    # and pair k rotates the elements k and k + head_dim / 2. Anima stretches the height and width bands
    # with an NTK factor of 4. Images have one frame, so the time band never rotates.
    def __init__(self, head_dim: int = 128, spatial_extrapolation: float = 4.0, base: float = 10000.0):
        dim_spatial = head_dim // 6 * 2
        dim_temporal = head_dim - 2 * dim_spatial
        spatial_base = base * spatial_extrapolation ** (dim_spatial / (dim_spatial - 2))
        self._num_temporal = dim_temporal // 2
        self._spatial = 1.0 / (spatial_base ** (mx.arange(0, dim_spatial, 2, dtype=mx.float32) / dim_spatial))

    def __call__(self, height: int, width: int) -> tuple[mx.array, mx.array]:
        # cos and sin per token in (h, w) order, float32, shape (height * width, head_dim / 2)
        h = mx.broadcast_to((mx.arange(height, dtype=mx.float32)[:, None] * self._spatial)[:, None], (height, width, self._spatial.shape[0]))  # fmt: off
        w = mx.broadcast_to((mx.arange(width, dtype=mx.float32)[:, None] * self._spatial)[None], (height, width, self._spatial.shape[0]))  # fmt: off
        t = mx.zeros((height, width, self._num_temporal), dtype=mx.float32)
        angles = mx.concatenate([t, h, w], axis=-1).reshape(height * width, -1)
        return mx.cos(angles), mx.sin(angles)

    @staticmethod
    def rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x: (B, L, heads, head_dim). The reference rotates in float32 and casts back.
        x32 = x.astype(mx.float32)
        a, b = mx.split(x32, 2, axis=-1)
        cos, sin = cos[:, None, :], sin[:, None, :]
        return mx.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1).astype(x.dtype)
