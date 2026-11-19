"""3b-2 harness: V0 split layers with enumerated tiers and W8A8 pre support.

Same structure as split_layers.py (column prefix on the ANE, GPU suffix, one
concat join) with three changes measured here:
  - ANE models are the enumerated production builders (ane_models.py), rows are
    padded up to pick_tier(T) and outputs sliced back;
  - the GPU half matmuls only the true rows (V0 paid for padded rows too);
  - precision selects the W8A16 zero-copy path or the W8A8 pre path
    (host quantise -> ext.ane_linear_q8).
Weights come from the instantiated transformer; packages cache by content hash.
"""

import hashlib
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPIKE / "ane_linear"))
import _ane_linear as ext  # noqa: E402
import ane_models as AM  # noqa: E402
from split_layers import ATTN_V_HEADROOM, W2_HEADROOM, _np32, _round64  # noqa: E402

TIERS = [4608, 6528]


def ane_model_for(w: np.ndarray, prec: str, ksplit: int, tag: str, cache_dir: Path, tiers=None) -> ext.AneModel:
    """Core ML package for weight prefix w[oa, I], built once per content hash."""
    tiers = tiers or TIERS
    O, I = w.shape
    digest = hashlib.sha1(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest()[:12]
    path = cache_dir / f"{tag}_{prec}_{O}x{I}_k{ksplit}_{digest}.mlpackage"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        wq, sc = AM.quantize_perchannel(w)
        if prec == "w8a16":
            prog, ct_in = AM.build_enum_w8a16(wq, sc, ksplit=ksplit, tiers=tiers)
        elif prec == "w8a8":
            prog, ct_in = AM.build_enum_q8pre(wq, sc, ksplit=ksplit, tiers=tiers)
        else:
            raise ValueError(f"unknown precision {prec}")
        AM.convert(prog, ct_in, path)
    return ext.AneModel(str(path), max(tiers), I, O)


class _AneMixin:
    DRAIN = True  # mx.eval the ANE input before dispatch; mx.compile forbids it, set False there

    def _launch_ane(self, xp, xf):
        """xp = input padded to the tier, xf = true-row input. Returns (hack_src, y_full).

        y_full is the ANE output cast FULL WIDTH to xp's dtype before any slicing.
        Measured on this MLX (block_q8 bisect): the GPU half must consume xp (the
        shared padded array) and the cast must precede row slicing, or the GPU
        suffix waits for the ANE call instead of overlapping it (55 ms vs 32 ms
        at w13). Slicing a cast array stays a view and is free. The primitive's
        input is drained with mx.eval before the call: an unmaterialized quantise
        chain makes the CPU-stream dispatch stall the whole graph (54 vs 34 ms).
        """
        T = xf.shape[0]
        if self.prec == "w8a16":
            xa = (xp * (1.0 / self.headroom)).astype(mx.float16) if self.headroom > 1.0 else xp.astype(mx.float16)
            if self.DRAIN:
                mx.eval(xa)
            ya = ext.ane_linear(xa, self.ane)
            return xa, ya.astype(xp.dtype)
        x32 = xf.astype(mx.float32)
        am = mx.max(mx.abs(x32), axis=1, keepdims=False)
        tok = mx.maximum(am / 127.0, mx.array(1e-6))
        q = mx.clip(mx.round(x32 / tok[:, None]), -127, 127).astype(mx.int8)
        # The model multiplies its output by tok, so scale tok itself by
        # 1/headroom: quantise grid unchanged, fp16 model output stays in
        # range, and _scale undoes the headroom like the W8A16 path.
        tok = tok * (1.0 / self.headroom)
        pad = xp.shape[0] - T
        if pad:
            q = mx.pad(q, [(0, pad), (0, 0)])
            tok = mx.pad(tok, [(0, pad)])
        if self.DRAIN:
            mx.eval(q, tok)
        ya = ext.ane_linear_q8(q, tok, self.ane)
        return q, ya.astype(xp.dtype)

    def _gpu_input(self, xp, xa):
        if self.order_hack and self.ane is not None:
            return xp + (xa[:1, :1] * 0).astype(xp.dtype)
        return xp


class SplitLinear(_AneMixin, nn.Module):
    """One linear: columns [0:oa) on the ANE, [oa:O) on the GPU, one concat join."""

    def __init__(self, w, gpu_share, cache_dir, tag, ksplit=1, headroom=1.0, prec="w8a16"):
        super().__init__()
        O, I = w.shape
        og = min(max(_round64(O * gpu_share), 0), O)
        self.oa, self.I, self.headroom, self.prec = O - og, I, headroom, prec
        self.tiers = TIERS
        self.order_hack = True
        self.wg = mx.array(w[self.oa :]).astype(mx.bfloat16) if og > 0 else None
        self.ane = ane_model_for(w[: self.oa], prec, ksplit, tag, cache_dir) if self.oa > 0 else None
        if headroom > 1.0:
            self._scale = mx.concatenate([mx.full((self.oa,), 1.0 * headroom), mx.ones((O - self.oa,))])
        else:
            self._scale = None

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, self.I)
        T = xf.shape[0]
        if self.ane is None:
            y = mx.matmul(xf, self.wg.T.astype(xf.dtype))
            return y.astype(x.dtype).reshape(*shp[:-1], self.wg.shape[0])
        tier = AM.pick_tier(T, self.tiers)
        xp = mx.pad(xf, [(0, tier - T), (0, 0)]) if tier > T else xf
        xa, y_full = self._launch_ane(xp, xf)
        gx = self._gpu_input(xp, xa)
        parts = [y_full[:T, : self.oa]]
        if self.wg is not None:
            parts.append(mx.matmul(gx, self.wg.T.astype(xp.dtype))[:T])
        y = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
        if self._scale is not None:
            y = y * self._scale.astype(y.dtype)
        return y.astype(x.dtype).reshape(*shp[:-1], self.oa + (0 if self.wg is None else self.wg.shape[0]))


class FusedSplit(_AneMixin, nn.Module):
    """Projections sharing one input; one ANE call over the fused prefix, per-projection suffixes."""

    def __init__(self, ws, gpu_share, cache_dir, tag, headroom=1.0, prec="w8a16"):
        super().__init__()
        Os = [w.shape[0] for w in ws]
        I = ws[0].shape[1]
        O = sum(Os)
        og = min(max(_round64(O * gpu_share), 0), O)
        self.oa, self.I, self.headroom, self.prec = O - og, I, headroom, prec
        self.tiers = TIERS
        self.order_hack = True
        self.a_p, self.gpu_w = [], []
        off = 0
        for d_p, w_p in zip(Os, ws, strict=True):
            a = max(0, min(d_p, self.oa - off))
            self.a_p.append(a)
            self.gpu_w.append(mx.array(w_p[a:]).astype(mx.bfloat16) if d_p - a > 0 else None)
            off += d_p
        self.offs = np.cumsum([0] + Os[:-1]).tolist()
        self.ane = None
        if self.oa > 0:
            w_ane = np.concatenate([ws[i][: self.a_p[i]] for i in range(len(ws))], axis=0)
            self.ane = ane_model_for(w_ane, prec, 1, tag, cache_dir)

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, self.I)
        T = xf.shape[0]
        xa, y_full = None, None
        xp = xf
        if self.ane is not None:
            tier = AM.pick_tier(T, self.tiers)
            xp = mx.pad(xf, [(0, tier - T), (0, 0)]) if tier > T else xf
            xa, y_full = self._launch_ane(xp, xf)
        outs = []
        for i, w in enumerate(self.gpu_w):
            parts = []
            a = self.a_p[i]
            if a > 0:
                parts.append(y_full[:T, self.offs[i] : self.offs[i] + a])
            if w is not None:
                gx = self._gpu_input(xp, xa)
                parts.append(mx.matmul(gx, w.T.astype(xp.dtype))[:T])
            yf = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
            outs.append(yf.astype(x.dtype).reshape(*shp[:-1], a + (0 if w is None else w.shape[0])))
        return outs


class AttentionSplit(nn.Module):
    """ZImageAttention with qkv fused split and split output projection."""

    def __init__(self, attn, cache_dir, block, prec="w8a16", gpu_share=0.5):
        super().__init__()
        ws = [_np32(attn.to_q.weight), _np32(attn.to_k.weight), _np32(attn.to_v.weight)]
        self.qkv = FusedSplit(ws, gpu_share, cache_dir, f"b{block}_qkv_{prec}", prec=prec)
        self.out = SplitLinear(_np32(attn.to_out[0].weight), gpu_share, cache_dir, f"b{block}_o_{prec}", prec=prec)
        self.a = attn

    def set_order_hack(self, v):
        self.qkv.order_hack = v
        self.out.order_hack = v

    def __call__(self, hidden_states, attention_mask=None, freqs_cis=None):
        attn = self.a
        b, s, _ = hidden_states.shape
        q, k, v = self.qkv(hidden_states)
        v = v * (1.0 / ATTN_V_HEADROOM)
        q = q.reshape(b, s, attn.n_heads, attn.head_dim)
        k = k.reshape(b, s, attn.n_heads, attn.head_dim)
        v = v.reshape(b, s, attn.n_heads, attn.head_dim)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if freqs_cis is not None:
            q = attn.apply_rotary_emb(q, freqs_cis)
            k = attn.apply_rotary_emb(k, freqs_cis)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        mask = None
        if attention_mask is not None:
            mask = mx.where(
                attention_mask[:, None, None, :],
                mx.array(0.0, dtype=q.dtype),
                mx.array(float("-inf"), dtype=q.dtype),
            )
        from mlx.core.fast import scaled_dot_product_attention

        h = scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
        h = mx.transpose(h, axes=(0, 2, 1, 3)).reshape(b, s, attn.dim)
        return self.out(h)


class FeedForwardSplit(nn.Module):
    """FeedForward with fused w1w3 split and split w2 (K-split 4, headroom 32)."""

    def __init__(self, ff, cache_dir, block, prec="w8a16", gpu_share=0.5):
        super().__init__()
        ws = [_np32(ff.w1.weight), _np32(ff.w3.weight)]
        self.w13 = FusedSplit(ws, gpu_share, cache_dir, f"b{block}_w13_{prec}", prec=prec)
        self.w2 = SplitLinear(
            _np32(ff.w2.weight), gpu_share, cache_dir, f"b{block}_w2_{prec}", ksplit=4, headroom=W2_HEADROOM, prec=prec
        )

    def set_order_hack(self, v):
        self.w13.order_hack = v
        self.w2.order_hack = v

    def __call__(self, x):
        # Fixed 2026-11-19: the old code pre-scaled h3 by 1/32 but SplitLinear._scale
        # only restored the ANE columns, leaving the w2 GPU columns 32x small. The
        # headroom now lives entirely inside w2 (pre-scale + restore on its own ANE
        # columns), matching production FeedForward under bfloat16 stream.
        h1, h3 = self.w13(x)
        return self.w2(nn.silu(h1) * h3)
