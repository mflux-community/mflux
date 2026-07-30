import mlx.core as mx

# Source: pixeldit_official.py get_2d_sincos_pos_embed / get_1d_sincos_pos_embed_from_grid
# (lines 32-80). cls_token/extra_tokens/device args dropped -- unused by PixDiT's
# pixel-decoder callers, which only ever want a plain [H*W, D] grid.


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: mx.array) -> mx.array:
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = 1.0 / (10000 ** (mx.arange(half, dtype=mx.float32) / half))
    out = pos.reshape(-1, 1) * omega.reshape(1, -1)  # [M, half]
    return mx.concatenate([mx.sin(out), mx.cos(out)], axis=1)  # [M, embed_dim]


def get_2d_sincos_pos_embed(embed_dim: int, height: int, width: int | None = None) -> mx.array:
    """Standard ViT-style 2D sin/cos positional embedding. Returns [height*width, embed_dim]."""
    width = height if width is None else width
    grid_h = mx.arange(height, dtype=mx.float32)
    grid_w = mx.arange(width, dtype=mx.float32)
    # indexing="xy": grid_x varies along columns (width), grid_y along rows (height);
    # flattening row-major gives position index = h * width + w, matching the
    # [H, W, D] reshape callers do downstream.
    grid_x, grid_y = mx.meshgrid(grid_w, grid_h, indexing="xy")
    emb_x = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_x.reshape(-1))
    emb_y = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_y.reshape(-1))
    return mx.concatenate([emb_x, emb_y], axis=1)
