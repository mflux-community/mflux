"""Custom Metal kernels for the Qwen-Image-2.1 attention prologue.

`fused_qk_norm_rope` replaces, per Q and K projection: cast-to-fp32 -> RMSNorm
(with its own cast) -> cast-back -> RoPE (fp32 rotate) -> cast-back — i.e. ~6
eager kernels (2+ when mx.compiled) — with a single kernel that reads each
tensor once, keeps fp32 in registers, and writes each tensor once, handling Q
and K in the same launch.

Set MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE=1 to fall back to the composed-ops path.
"""

from __future__ import annotations

import os

import mlx.core as mx

_FUSED_QK_NORM_ROPE_SOURCE = """
    constexpr float eps = 1e-6;

    uint row = threadgroup_position_in_grid.x;
    uint i = thread_position_in_threadgroup.x;   // pair index: elements (2i, 2i+1)

    int L = q_shape[1];
    int H = q_shape[2];
    int D = q_shape[3];
    int hd = D / 2;

    int l = (row / H) % L;

    uint base = row * D;
    uint off = base + 2 * i;

    float aq = static_cast<float>(q[off]);
    float bq = static_cast<float>(q[off + 1]);
    float ak = static_cast<float>(k[off]);
    float bk = static_cast<float>(k[off + 1]);

    float sq = aq * aq + bq * bq;
    float sk = ak * ak + bk * bk;

    // reduce across this row's threadgroup (D/2 threads, full simdgroups)
    sq = simd_sum(sq);
    sk = simd_sum(sk);
    int nsg = simdgroups_per_threadgroup;
    threadgroup float shq[8];
    threadgroup float shk[8];
    if (thread_index_in_simdgroup == 0) {
        shq[simdgroup_index_in_threadgroup] = sq;
        shk[simdgroup_index_in_threadgroup] = sk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float totq = 0.0;
    float totk = 0.0;
    for (int g = 0; g < nsg; g++) {
        totq += shq[g];
        totk += shk[g];
    }
    float rrq = rsqrt(totq / D + eps);
    float rrk = rsqrt(totk / D + eps);

    float wqa = static_cast<float>(wq[2 * i]);
    float wqb = static_cast<float>(wq[2 * i + 1]);
    float wka = static_cast<float>(wk[2 * i]);
    float wkb = static_cast<float>(wk[2 * i + 1]);

    float nqa = aq * rrq * wqa;
    float nqb = bq * rrq * wqb;
    float nka = ak * rrk * wka;
    float nkb = bk * rrk * wkb;

    float c = static_cast<float>(cos_t[l * hd + i]);
    float s = static_cast<float>(sin_t[l * hd + i]);

    out_q[off] = T(nqa * c - nqb * s);
    out_q[off + 1] = T(nqa * s + nqb * c);
    out_k[off] = T(nka * c - nkb * s);
    out_k[off + 1] = T(nka * s + nkb * c);
"""

_fused_qk_norm_rope_kernel = None


def _get_kernel():
    global _fused_qk_norm_rope_kernel
    if _fused_qk_norm_rope_kernel is None:
        try:
            _fused_qk_norm_rope_kernel = mx.fast.metal_kernel(
                name="fused_qk_norm_rope",
                input_names=["q", "k", "wq", "wk", "cos_t", "sin_t"],
                output_names=["out_q", "out_k"],
                source=_FUSED_QK_NORM_ROPE_SOURCE,
            )
        except Exception:  # noqa: BLE001 — non-Metal hosts fall back silently
            _fused_qk_norm_rope_kernel = False
    return _fused_qk_norm_rope_kernel or None


def fused_qk_norm_rope_available(head_dim: int) -> bool:
    """The kernel needs full simdgroups (D/2 a multiple of threads-per-simdgroup)."""
    if os.environ.get("MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE"):
        return False
    if head_dim % 64 != 0:  # D/2 must be a multiple of 32
        return False
    if head_dim > 512:  # the shq/shk reduction buffers hold 8 simdgroups (D/2 <= 256 threads)
        return False
    return _get_kernel() is not None


def fused_qk_norm_rope(
    q_flat: mx.array,
    k_flat: mx.array,
    weight_q: mx.array,
    weight_k: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    num_heads: int,
    head_dim: int,
) -> tuple[mx.array, mx.array] | None:
    """RMSNorm + RoPE for Q and K in one pass.

    q_flat/k_flat: [B, L, H*D] contiguous (raw projection outputs); rope tables
    must cover exactly L positions. Returns the same [B, L, H*D] layout, or
    None when custom kernels are unavailable.
    """
    kernel = _get_kernel()
    if kernel is None:
        return None
    B, L, _ = q_flat.shape
    H, D = num_heads, head_dim
    # Fail safe on geometry mismatches (e.g. an axes_dims_rope whose per-axis
    # widths do not sum to head_dim): the composed path then raises the same
    # loud broadcast error as stock instead of silently reading wrong angles.
    if rope_cos.shape[-1] != D // 2 or rope_sin.shape[-1] != D // 2:
        return None
    if rope_cos.shape[0] != L or rope_sin.shape[0] != L:
        return None
    if rope_cos.dtype != mx.float32:
        rope_cos = rope_cos.astype(mx.float32)
    if rope_sin.dtype != mx.float32:
        rope_sin = rope_sin.astype(mx.float32)
    q = q_flat.reshape(B, L, H, D)
    k = k_flat.reshape(B, L, H, D)
    out_q, out_k = kernel(
        inputs=[q, k, weight_q, weight_k, rope_cos, rope_sin],
        template=[("T", q.dtype)],
        output_shapes=[(B, L, H, D), (B, L, H, D)],
        output_dtypes=[q.dtype, q.dtype],
        # NOTE: mx.fast.metal_kernel's `grid` is in THREADS (not threadgroups):
        # B*L*H rows x D/2 pair-threads per row.
        grid=(B * L * H * (D // 2), 1, 1),
        threadgroup=(D // 2, 1, 1),
    )
    return out_q.reshape(B, L, H * D), out_k.reshape(B, L, H * D)
