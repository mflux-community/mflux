import mlx.core as mx


class SenseNovaRope:
    # The head dim splits into t (64), h (32) and w (32); each segment rotates by its own index with rotate_half
    T_THETA = 5_000_000.0
    HW_THETA = 10_000.0
    SEGMENTS = (64, 32, 32)

    @staticmethod
    def indexes(text_len: int, grid_h: int, grid_w: int) -> mx.array:
        # (3, N) t/h/w indexes of the image tokens, which all share the t index text_len of their prefix
        idx = mx.arange(grid_h * grid_w, dtype=mx.int32)
        return mx.stack([mx.full(idx.shape, text_len, dtype=mx.int32), idx // grid_w, idx % grid_w])

    @staticmethod
    def text_indexes(text_len: int) -> mx.array:
        # (3, L) indexes of the prompt prefix: t counts the tokens, h and w stay 0
        idx = mx.arange(text_len, dtype=mx.int32)
        return mx.stack([idx, mx.zeros_like(idx), mx.zeros_like(idx)])

    @staticmethod
    def cos_sin(indexes: mx.array, dtype: mx.Dtype) -> list[tuple[mx.array, mx.array]]:
        # fp32 tables per axis, [1, N, 1, segment] in the activation dtype
        tables = []
        for axis, (dim, theta) in enumerate(zip(SenseNovaRope.SEGMENTS, (SenseNovaRope.T_THETA,) + (SenseNovaRope.HW_THETA,) * 2)):  # fmt: skip
            freqs = 1.0 / theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
            angles = indexes[axis].astype(mx.float32)[:, None] * freqs[None, :]
            emb = mx.concatenate([angles, angles], axis=-1)
            tables.append((mx.cos(emb)[None, :, None, :].astype(dtype), mx.sin(emb)[None, :, None, :].astype(dtype)))
        return tables

    @staticmethod
    def apply(x: mx.array, tables: list[tuple[mx.array, mx.array]]) -> mx.array:
        # x: [B, S, H, 128]
        out = []
        start = 0
        for dim, (cos, sin) in zip(SenseNovaRope.SEGMENTS, tables):
            seg = x[..., start : start + dim]
            half = dim // 2
            rotated = mx.concatenate([-seg[..., half:], seg[..., :half]], axis=-1)
            out.append(seg * cos + rotated * sin)
            start += dim
        return mx.concatenate(out, axis=-1)
