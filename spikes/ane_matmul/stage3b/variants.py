"""3b-1 assembly variants: partial-sum splits instead of column concatenation.

V0 (baseline, split_layers.py): column-prefix ANE + GPU suffix, concatenated.
E2 AttentionHeadsSplit: the ANE owns the first 15 heads of q, k, v as one fused
   call; GPU runs the other 15. Two SDPA calls (heads are independent), and the
   output projection is split by INPUT ROWS with partial outputs added. No
   wide concat anywhere on the attention path.
E1 FfRowsSplit (Megatron-style FF): the ANE owns the first 5120 hidden units of
   w1 AND w3 (one fused call), computes hidden_a, then a second ANE call over
   the first 5120 rows of w2 (K-split 2) produces the partial output. The GPU
   runs the other half; outputs are ADDED, never concatenated.

Both keep the ordering-workaround flag. Correctness target: per-block cosine
> 0.9999 against V0 on the same random weights.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SPIKE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPIKE / "ane_linear"))
sys.path.insert(0, str(SPIKE))
import _ane_linear  # noqa: E402
from ane_split import build_ane_model  # noqa: E402

T_FIXED = 4224
W2_HEADROOM = 32.0
V_HEADROOM = 4.0


def _np32(a):
    return np.array(a.astype(mx.float32) if isinstance(a, mx.array) else a).astype(np.float32)


def _pad(xf):
    T = xf.shape[0]
    return mx.pad(xf, [(0, T_FIXED - T), (0, 0)]) if T < T_FIXED else xf


class AttentionHeadsSplit(nn.Module):
    def __init__(self, attn: nn.Module, cache_dir: Path, block: int, heads_ane: int = 15):
        super().__init__()
        self.a = attn
        dim = attn.dim
        hd = attn.head_dim
        na = heads_ane * hd  # columns per projection on the ANE side
        q = _np32(attn.to_q.weight)
        k = _np32(attn.to_k.weight)
        v = _np32(attn.to_v.weight)
        w_ane = np.concatenate([q[:na], k[:na], v[:na]], axis=0)  # [3*na, dim]
        self.oa = w_ane.shape[0]
        path = build_ane_model(w_ane, cache_dir, f"b{block}_qkvh", ksplit=1)
        self.ane = _ane_linear.AneModel(path, T_FIXED, dim, self.oa)
        self.wq_b = mx.array(q[na:]).astype(mx.bfloat16)
        self.wk_b = mx.array(k[na:]).astype(mx.bfloat16)
        self.wv_b = mx.array(v[na:]).astype(mx.bfloat16)
        wo = _np32(attn.to_out[0].weight)
        self.wo_a = mx.array(wo[:, :na]).astype(mx.bfloat16)
        self.wo_b = mx.array(wo[:, na:]).astype(mx.bfloat16)
        self.na, self.hd, self.dim = na, hd, dim
        self.heads_ane = heads_ane
        self.order_hack = True

    def __call__(self, hidden_states, attention_mask=None, freqs_cis=None):
        attn = self.a
        b, s, _ = hidden_states.shape
        xf = hidden_states.reshape(-1, self.dim)
        T = xf.shape[0]
        xp = _pad(xf)
        xa = xp.astype(mx.float16)
        ya = _ane_linear.ane_linear(xa, self.ane).astype(mx.bfloat16)
        gx = xp + (xa[:1, :1] * 0).astype(xp.dtype) if self.order_hack else xp
        gx = gx[:T]
        hz = (b, s, -1, self.hd)
        yT = ya[:T]
        qA = yT[:, : self.na].reshape(*hz)
        kA = yT[:, self.na : 2 * self.na].reshape(*hz)
        vA = (yT[:, 2 * self.na : 3 * self.na] * (1.0 / V_HEADROOM)).reshape(*hz)
        qB = mx.matmul(gx, self.wq_b.T).reshape(*hz)
        kB = mx.matmul(gx, self.wk_b.T).reshape(*hz)
        vB = (mx.matmul(gx, self.wv_b.T) * (1.0 / V_HEADROOM)).reshape(*hz)
        if attn.norm_q is not None:
            qA, qB = attn.norm_q(qA), attn.norm_q(qB)
        if attn.norm_k is not None:
            kA, kB = attn.norm_k(kA), attn.norm_k(kB)
        if freqs_cis is not None:
            qA = attn.apply_rotary_emb(qA, freqs_cis)
            kA = attn.apply_rotary_emb(kA, freqs_cis)
            qB = attn.apply_rotary_emb(qB, freqs_cis)
            kB = attn.apply_rotary_emb(kB, freqs_cis)

        mask = None
        if attention_mask is not None:
            mask = mx.where(
                attention_mask[:, None, None, :],
                mx.array(0.0, dtype=qA.dtype),
                mx.array(float("-inf"), dtype=qA.dtype),
            )

        def sdpa(q, k, v):
            from mlx.core.fast import scaled_dot_product_attention

            q = mx.transpose(q, axes=(0, 2, 1, 3))
            k = mx.transpose(k, axes=(0, 2, 1, 3))
            v = mx.transpose(v, axes=(0, 2, 1, 3))
            h = scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
            return mx.transpose(h, axes=(0, 2, 1, 3)).reshape(b, s, -1)

        ctxA = sdpa(qA, kA, vA)
        ctxB = sdpa(qB, kB, vB)
        return mx.matmul(ctxA, self.wo_a.T) + mx.matmul(ctxB, self.wo_b.T)

    def set_order_hack(self, v):
        self.order_hack = v


class FfRowsSplit(nn.Module):
    def __init__(self, ff: nn.Module, cache_dir: Path, block: int, half: int = 5120):
        super().__init__()
        dim = ff.w1.weight.shape[1]
        w1 = _np32(ff.w1.weight)
        w3 = _np32(ff.w3.weight)
        w2 = _np32(ff.w2.weight)
        self.half = half
        w_ane1 = np.concatenate([w1[:half], w3[:half]], axis=0)  # [2*half, dim]
        p1 = build_ane_model(w_ane1, cache_dir, f"b{block}_w13r", ksplit=1)
        self.ane1 = _ane_linear.AneModel(p1, T_FIXED, dim, w_ane1.shape[0])
        w2a = w2[:, :half]
        p2 = build_ane_model(w2a, cache_dir, f"b{block}_w2a", ksplit=2)
        self.ane2 = _ane_linear.AneModel(p2, T_FIXED, half, w2a.shape[0])
        self.w1_b = mx.array(w1[half:]).astype(mx.bfloat16)
        self.w3_b = mx.array(w3[half:]).astype(mx.bfloat16)
        self.w2_b = mx.array(w2[:, half:]).astype(mx.bfloat16)
        self.dim = dim
        self.order_hack = True

    def _call_ane(self, xp, model):
        xa = xp.astype(mx.float16)
        return xa, _ane_linear.ane_linear(xa, model)

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, self.dim)
        T = xf.shape[0]
        xp = _pad(xf)
        gx = xp + (xp[:1, :1] * 0).astype(xp.dtype) if self.order_hack else xp
        # ANE hidden units: fused w1a|w3a, SwiGLU, then w2a as a second ANE call.
        xa1 = xp.astype(mx.float16)
        ya1 = _ane_linear.ane_linear(xa1, self.ane1).astype(mx.bfloat16)
        h1a = ya1[:, : self.half]
        h3a = ya1[:, self.half :] * (1.0 / W2_HEADROOM)
        hidden_a = nn.silu(h1a) * h3a
        xa2 = (hidden_a * (1.0 / W2_HEADROOM)).astype(mx.float16)
        ya2 = _ane_linear.ane_linear(xa2, self.ane2)
        out_a = (ya2.astype(mx.float32) * W2_HEADROOM).astype(mx.bfloat16)
        # GPU hidden units.
        h1b = mx.matmul(gx, self.w1_b.T)
        h3b = mx.matmul(gx, self.w3_b.T) * (1.0 / W2_HEADROOM)
        hidden_b = nn.silu(h1b) * h3b
        out_b = mx.matmul(hidden_b, self.w2_b.T)
        return (out_a + out_b)[:T].astype(x.dtype).reshape(*shp[:-1], out_a.shape[-1])

    def set_order_hack(self, v):
        self.order_hack = v
