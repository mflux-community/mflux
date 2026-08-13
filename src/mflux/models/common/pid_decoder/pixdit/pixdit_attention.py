import mlx.core as mx

# MLX only takes its *fused* attention kernel (never materializing the [B, H, S, S] scores)
# for these head_dims; anything else silently falls back to the dense implementation.
# Measured on this machine at S=8192/H=16: head_dim 64 -> 0.13GB, 80 -> 0.16GB, but
# 72 -> 4.18GB, i.e. exactly the dense score matrix.
_FLASH_HEAD_DIMS = (64, 80, 128)


def flash_sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float, mask=None) -> mx.array:
    """Scaled dot-product attention that stays on MLX's fused kernel for any head_dim.

    PidNet's pixel stream runs head_dim=72 (attn 1152 / 16 heads), which is not a fused-kernel
    size, so a full-resolution decode materializes the dense scores: at a 3328x4992 output that
    is 64896^2 * 16 * 4B = 269GB, and the decode dies in metal::malloc. Zero-padding q/k/v's
    head_dim up to the next supported size keeps the fused path.

    Exact, not an approximation: the padded lanes are zero in both q and k, so every extra
    product in QK^T is zero and the scores are unchanged; `scale` stays the caller's original
    head_dim^-0.5. v's padded lanes only produce padded output lanes, which are sliced back off.
    A head_dim already in the fused set (the patch stream's 64) takes the direct path.
    """
    head_dim = q.shape[-1]
    target = next((d for d in _FLASH_HEAD_DIMS if d >= head_dim), None)
    if target is None or target == head_dim:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    pad_width = [(0, 0)] * (q.ndim - 1) + [(0, target - head_dim)]
    q, k, v = (mx.pad(t, pad_width) for t in (q, k, v))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return out[..., :head_dim]
