"""Custom Metal kernels for the Qwen-Image-2.1 attention prologue.

`fused_qk_norm_rope` combines native RMSNorm rounding, FP32 RoPE, and the
sequence-major to head-major layout change in one Q/K kernel. Projection,
norm-weight, and rotary-table views are read using their original strides.

Set MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE=1 to fall back to the composed-ops path.
"""

from __future__ import annotations

import os

import mlx.core as mx

_FUSED_QK_NORM_ROPE_SOURCE = """
    constexpr float eps = EPS_VALUE;
    constexpr uint blocks = (D + 127) / 128;
    constexpr uint pairs_per_lane = 2 * blocks;
    uint row = thread_position_in_grid.x / 32;
    if (row >= B * L * H) return;
    uint lane = thread_index_in_simdgroup;
    uint h = row % H;
    uint l = (row / H) % L;
    uint b = row / (H * L);
    size_t qb = b * q_strides[0] + l * q_strides[1] + h * D * q_strides[2];
    size_t kb = b * k_strides[0] + l * k_strides[1] + h * D * k_strides[2];

    float qv[4 * blocks], kv[4 * blocks];
    float sq = 0.0f, sk = 0.0f;
    // Match native RMSNorm: four adjacent values per lane, then subgroup totals.
    for (uint block = 0; block < blocks; ++block) {
        float bq = 0.0f, bk = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            uint c = block * 128 + lane * 4 + j;
            uint i = block * 4 + j;
            qv[i] = c < D ? float(TQ(q[qb + c * q_strides[2]])) : 0.0f;
            kv[i] = c < D ? float(TK(k[kb + c * k_strides[2]])) : 0.0f;
            bq += qv[i] * qv[i];
            bk += kv[i] * kv[i];
        }
        bq = simd_sum(bq);
        bk = simd_sum(bk);
        if (blocks == 1 || lane == block) {
            sq = bq;
            sk = bk;
        }
    }
    float rrq = metal::precise::rsqrt((blocks == 1 ? sq : simd_sum(sq)) / float(D) + eps);
    float rrk = metal::precise::rsqrt((blocks == 1 ? sk : simd_sum(sk)) / float(D) + eps);
    size_t out = ((size_t(b) * H + h) * L + l) * D;

    for (uint pair = 0; pair < pairs_per_lane; ++pair) {
        #pragma STDC FP_CONTRACT OFF
        // Composed RoPE rounds the products before its separate add/subtract.
        uint c = (pair / 2) * 128 + lane * 4 + (pair % 2) * 2;
        if (c >= D) continue;
        uint p = c / 2;
        // MLX RMSNorm promotes x/weight together, rounds normalized x, then
        // rounds the weighted result before the FP32 rotary arithmetic.
        float nqa = float(TQ(float(TQ(qv[2 * pair] * rrq)) * float(TQ(wq[c * wq_strides[0]]))));
        float nqb = float(TQ(float(TQ(qv[2 * pair + 1] * rrq)) * float(TQ(wq[(c + 1) * wq_strides[0]]))));
        float nka = float(TK(float(TK(kv[2 * pair] * rrk)) * float(TK(wk[c * wk_strides[0]]))));
        float nkb = float(TK(float(TK(kv[2 * pair + 1] * rrk)) * float(TK(wk[(c + 1) * wk_strides[0]]))));
        float cosine = float(cos_t[l * cos_t_strides[0] + p * cos_t_strides[1]]);
        float sine = float(sin_t[l * sin_t_strides[0] + p * sin_t_strides[1]]);
        out_q[out + c] = TQ(nqa * cosine - nqb * sine);
        out_q[out + c + 1] = TQ(nqa * sine + nqb * cosine);
        out_k[out + c] = TK(nka * cosine - nkb * sine);
        out_k[out + c + 1] = TK(nka * sine + nkb * cosine);
    }
"""

_fused_qk_norm_rope_kernels: dict[float, object] = {}
_fused_qk_norm_rope_kernel = None  # eps 1e-6 kernel, kept for the Qwen tests
_fused_qk_norm_rope_unavailable = False


def _get_kernel(eps: float = 1e-6):
    global _fused_qk_norm_rope_kernel, _fused_qk_norm_rope_unavailable
    eps = float(eps)  # an int such as 0 would render as the invalid Metal literal "0f"
    if _fused_qk_norm_rope_unavailable or not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return None
    if eps != 1e-6:
        kernel = _fused_qk_norm_rope_kernels.get(eps)
        if kernel is None:
            try:
                kernel = mx.fast.metal_kernel(
                    name=f"fused_qk_norm_rope_eps_{eps!r}".replace(".", "p").replace("-", "m").replace("+", ""),
                    input_names=["q", "k", "wq", "wk", "cos_t", "sin_t"],
                    output_names=["out_q", "out_k"],
                    source=_FUSED_QK_NORM_ROPE_SOURCE.replace("EPS_VALUE", f"{eps!r}f"),
                    ensure_row_contiguous=False,
                    compile_options={"math_mode": "safe"},
                )
            except (RuntimeError, ValueError):
                _fused_qk_norm_rope_unavailable = True
                return None
            _fused_qk_norm_rope_kernels[eps] = kernel
        return kernel
    if _fused_qk_norm_rope_kernel is None:
        try:
            _fused_qk_norm_rope_kernel = mx.fast.metal_kernel(
                name="fused_qk_norm_rope",
                input_names=["q", "k", "wq", "wk", "cos_t", "sin_t"],
                output_names=["out_q", "out_k"],
                source=_FUSED_QK_NORM_ROPE_SOURCE.replace("EPS_VALUE", "1e-6f"),
                ensure_row_contiguous=False,
                compile_options={"math_mode": "safe"},
            )
        except (RuntimeError, ValueError):
            # Cache unsupported construction, not failures while dispatching the kernel.
            _fused_qk_norm_rope_unavailable = True
            return None
    return _fused_qk_norm_rope_kernel


def fused_qk_norm_rope_available(
    head_dim: int, eps: float = 1e-6, disable_env: str = "MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE"
) -> bool:
    """Supported head widths, with one SIMD group per normalization row."""
    if os.environ.get(disable_env):
        return False
    if head_dim <= 0 or head_dim % 64 != 0 or head_dim > 512:
        return False
    return _get_kernel(eps) is not None


def fused_qk_norm_rope(
    q_flat: mx.array,
    k_flat: mx.array,
    weight_q: mx.array,
    weight_k: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    disable_env: str = "MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE",
) -> tuple[mx.array, mx.array] | None:
    """RMSNorm + RoPE for Q and K in one pass.

    q_flat/k_flat: [B, L, H*D], including strided projection views; rope tables
    must cover exactly L positions. Returns head-major [B, H, L, D] arrays
    with each native RMSNorm's promoted dtype. Epsilon defaults to 1e-6.
    Returns None for unsupported inputs so callers can use composed ops.
    """
    if not fused_qk_norm_rope_available(head_dim, eps, disable_env):
        return None
    if q_flat.ndim != 3 or k_flat.shape != q_flat.shape:
        return None
    B, L, width = q_flat.shape
    H, D = num_heads, head_dim
    if B == 0 or L == 0 or H <= 0 or width != H * D:
        return None
    if weight_q.shape != (D,) or weight_k.shape != (D,):
        return None
    if rope_cos.shape != (L, D // 2) or rope_sin.shape != (L, D // 2):
        return None
    supported_dtypes = (mx.float16, mx.bfloat16, mx.float32)
    if any(x.dtype not in supported_dtypes for x in (q_flat, k_flat, weight_q, weight_k, rope_cos, rope_sin)):
        return None
    # fast.rms_norm promotes input and weight before normalization (mlx/fast.cpp).
    dtype_q = mx.result_type(q_flat, weight_q)
    dtype_k = mx.result_type(k_flat, weight_k)
    kernel = _get_kernel(eps)
    out_q, out_k = kernel(
        inputs=[q_flat, k_flat, weight_q, weight_k, rope_cos, rope_sin],
        template=[("TQ", dtype_q), ("TK", dtype_k), ("B", B), ("L", L), ("H", H), ("D", D)],
        output_shapes=[(B, H, L, D), (B, H, L, D)],
        output_dtypes=[dtype_q, dtype_k],
        grid=(B * L * H * 32, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out_q, out_k
