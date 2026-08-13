from functools import lru_cache

import mlx.core as mx


# The tables depend only on the grid, so every block and every sampling step asks for the same
# ones: at 4096x4096 that was 12 rebuilds of a [65536, dim] cos/sin pair per decode, ~38 MB
# each. The reference caches them too (PixDiT_T2I.precompute_pos / PiTBlock._pos_cache).
# maxsize caps what the cache pins: two entries per resolution (head_dim 64 patch stream,
# 72 pixel stream), so 4 covers a decode and bounds retention at ~150 MB.
@lru_cache(maxsize=4)
def precompute_freqs_cis_2d_ntk(
    dim: int,
    height: int,
    width: int,
    ref_grid_h: int,
    ref_grid_w: int,
    theta: float = 10000.0,
    scale: float = 16.0,
) -> tuple[mx.array, mx.array]:
    dim_axis = dim // 2
    h_scale = height / ref_grid_h
    w_scale = width / ref_grid_w
    h_ntk = h_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    w_ntk = w_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    h_theta = theta * h_ntk
    w_theta = theta * w_ntk

    x_pos = mx.linspace(0, scale, width)
    y_pos = mx.linspace(0, scale, height)
    y_pos, x_pos = mx.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)

    exponent = mx.arange(0, dim, 4)[: dim // 4].astype(mx.float32) / dim
    freqs_w = 1.0 / (w_theta**exponent)
    freqs_h = 1.0 / (h_theta**exponent)

    x_freqs = mx.outer(x_pos, freqs_w)
    y_freqs = mx.outer(y_pos, freqs_h)
    all_freqs = mx.stack([x_freqs, y_freqs], axis=-1).reshape(height * width, -1)
    return mx.cos(all_freqs), mx.sin(all_freqs)


def precompute_freqs_cis_1d_text(dim: int, length: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
    """1D RoPE freqs for the text stream. Source: PixDiT_T2I.fetch_pos_text
    (pixeldit_official.py:1345-1356) -- same head_dim as the image RoPE, applied
    per text-token position rather than per 2D grid location."""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    positions = mx.arange(0, length).astype(mx.float32)[:, None]
    angles = positions * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def apply_rotary_emb(xq: mx.array, xk: mx.array, freqs_cis: tuple[mx.array, mx.array]) -> tuple[mx.array, mx.array]:
    cos_freqs, sin_freqs = freqs_cis
    cos_freqs = cos_freqs[None, :, None, :]
    sin_freqs = sin_freqs[None, :, None, :]

    def _rotate(x):
        pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        xr, xi = pairs[..., 0], pairs[..., 1]
        out = mx.stack([xr * cos_freqs - xi * sin_freqs, xr * sin_freqs + xi * cos_freqs], axis=-1)
        return out.reshape(*x.shape[:-1], -1).astype(x.dtype)

    return _rotate(xq), _rotate(xk)
