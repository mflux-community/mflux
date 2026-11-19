"""Stage 3c weightless split layers: column prefix on the ANE via WLModel.

Same V0 rules the 3b-2 bisect forced (shared padded GPU input, full-width
cast before slicing, drain before the primitive in eager mode), with three
changes:
  - the program is weightless: it is built once per (Mpad, N, K), shared by
    all layers of that shape, and holds no scales;
  - weights are int8 codes the bridge wraps zero-copy, so no per-call weight
    copy exists and no double-buffering is needed;
  - the epilogue (transpose, token scale, channel scale) runs in MLX in fp32.
Design follows ~/zCode_26_11/drawthings-ane-research/DRAWTHINGS-ANE-FINDINGS.md
(liuliu/ccv, BSD-3-Clause); written from the description, no source copied.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ane_linear"))
sys.path.insert(0, str(HERE))
import _ane_linear as ext  # noqa: E402
import wl_programs  # noqa: E402


def align256(rows: int) -> int:
    return max(256, ((rows + 255) // 256) * 256)


def quantize_prefix(w_np: np.ndarray):
    """fp32 (N, K) -> (int8 codes MLX, fp32 per-channel scale x K MLX)."""
    sc = np.maximum(np.abs(w_np).max(axis=1) / 127.0, 1e-12).astype(np.float32)
    codes = np.clip(np.round(w_np / sc[:, None]), -127, 127).astype(np.int8)
    return mx.array(codes), mx.array(sc * w_np.shape[1])


class _WLBase(nn.Module):
    def __init__(self, I: int, oa: int, Mpad: int, compiled_ok: bool = True):
        super().__init__()
        self.I, self.oa, self.Mpad = I, oa, Mpad
        self.drain = not compiled_ok  # mx.compile forbids the inner eval
        self.order_hack = True

    def set_model(self, w_codes: mx.array, swK: mx.array):
        self.codes = w_codes
        self.swK = swK
        self.model = wl_programs.get_model(ext, self.Mpad, self.oa, self.I)

    def _launch(self, xp: mx.array):
        """Quantise padded rows, transpose into (I, Mpad), call the program.

        Returns (zero-source for the ordering hack, raw y fp16 (oa, Mpad)).
        """
        x32 = xp.astype(mx.float32)
        tx = mx.maximum(mx.max(mx.abs(x32), axis=1) / 127.0, mx.array(1e-8))
        q = mx.clip(mx.round(x32 / tx[:, None]), -127, 127).astype(mx.int8)
        xt = mx.reshape(mx.transpose(q, (1, 0)), (self.I * self.Mpad,))
        xt = mx.reshape(xt, (self.I, self.Mpad))
        if self.drain:
            mx.eval(xt)
        y = ext.wl_linear(xt, self.codes, self.model)  # (oa, Mpad) raw dot/K
        return tx, xt, y

    def _epilogue(self, y, tx, T):
        yy = mx.transpose(y, (1, 0))[:T]
        return yy.astype(mx.float32) * tx[:T, None] * self.swK[None, :]


class WLSplitLinear(_WLBase):
    """One projection: columns [0:oa) on the ANE weightless program, [oa:O) on the GPU suffix."""

    def __init__(self, w_np, gpu_suffix, oa, I, Mpad, compiled_ok=True):
        super().__init__(I, oa, Mpad, compiled_ok)
        self.gpu = gpu_suffix
        self.O = w_np.shape[0]
        codes, swK = quantize_prefix(np.ascontiguousarray(w_np[:oa]))
        self.set_model(codes, swK)

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, self.I)
        T = xf.shape[0]
        if T > self.Mpad:
            raise ValueError(f"tokens {T} exceed program Mpad {self.Mpad}")
        xp = mx.pad(xf, [(0, self.Mpad - T), (0, 0)]) if T < self.Mpad else xf
        tx, xt, y = self._launch(xp)
        parts = [self._epilogue(y, tx, T).astype(x.dtype)]
        if self.O > self.oa:
            gx = xp + (mx.reshape(xt, (-1,))[:1] * 0).astype(xp.dtype).reshape(1, 1) if self.order_hack else xp
            parts.append(self.gpu(gx)[:T])
        out = mx.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
        return out.astype(x.dtype).reshape(*shp[:-1], self.O)


class WLFusedSplit(_WLBase):
    """Projections sharing one input; one program over the stacked prefixes."""

    def __init__(self, ws_np, gpus, oa_list, I, Mpad, compiled_ok=True):
        oa = sum(oa_list)
        super().__init__(I, oa, Mpad, compiled_ok)
        self.ws, self.gpus, self.oa_list = ws_np, gpus, oa_list
        stack = np.concatenate([np.asarray(w[:a]) for w, a in zip(ws_np, oa_list, strict=True)], axis=0)
        codes, swK = quantize_prefix(np.ascontiguousarray(stack))
        self.set_model(codes, swK)
        self.offs = [0] * len(oa_list)
        off = 0
        for i, a in enumerate(oa_list):
            self.offs[i] = off
            off += a

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, self.I)
        T = xf.shape[0]
        xp = mx.pad(xf, [(0, self.Mpad - T), (0, 0)]) if T < self.Mpad else xf
        tx, xt, y = self._launch(xp)
        full = self._epilogue(y, tx, T)
        outs = []
        for i, (w, a, g) in enumerate(zip(self.ws, self.oa_list, self.gpus, strict=True)):
            parts = [full[:, self.offs[i] : self.offs[i] + a]]
            if g is not None:
                gx = xp + (mx.reshape(xt, (-1,))[:1] * 0).astype(xp.dtype).reshape(1, 1) if self.order_hack else xp
                parts.append(g(gx)[:T])
            yf = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
            outs.append(yf.astype(x.dtype).reshape(*shp[:-1], w.shape[0]))
        return outs


def _round64(v: float) -> int:
    return max(0, int(round(v / 64)) * 64)


def _prefix_split(dims, share):
    """Total output columns split once; per projection, ANE takes the prefix."""
    total = sum(dims)
    og = min(max(_round64(total * share), 0), total)
    oa = total - og
    counts, off = [], 0
    for d in dims:
        a = max(0, min(d, oa - off))
        counts.append(a)
        off += d
    return counts


def _suffix_module(w_np, a):
    if w_np.shape[0] - a <= 0:
        return None
    sl = nn.Linear(w_np.shape[1], w_np.shape[0] - a, bias=False)
    sl.weight = mx.array(w_np[a:]).astype(mx.bfloat16)
    return sl


class WLAttentionSplit(nn.Module):
    def __init__(self, attn, Mpad, share=0.5, compiled_ok=True):
        super().__init__()
        ws = [
            np.array(attn.to_q.weight.astype(mx.float32)),
            np.array(attn.to_k.weight.astype(mx.float32)),
            np.array(attn.to_v.weight.astype(mx.float32)),
        ]
        counts = _prefix_split([w.shape[0] for w in ws], share)
        gpus = [_suffix_module(w, a) for w, a in zip(ws, counts, strict=True)]
        self.qkv = WLFusedSplit(ws, gpus, counts, ws[0].shape[1], Mpad, compiled_ok)
        wo = np.array(attn.to_out[0].weight.astype(mx.float32))
        oa_o = wo.shape[0] - _round64(wo.shape[0] * share)
        self.out = WLSplitLinear(wo, _suffix_module(wo, oa_o), oa_o, wo.shape[1], Mpad, compiled_ok)
        self.a = attn

    def set_hack(self, v):
        self.qkv.order_hack = v
        self.out.order_hack = v

    def __call__(self, hidden_states, attention_mask=None, freqs_cis=None):
        attn = self.a
        b, s, _ = hidden_states.shape
        q, k, v = self.qkv(hidden_states)
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

        out = scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
        out = mx.transpose(out, axes=(0, 2, 1, 3)).reshape(b, s, attn.dim)
        return self.out(out)


class WLFeedForwardSplit(nn.Module):
    def __init__(self, ff, Mpad, share=0.5, compiled_ok=True):
        super().__init__()
        ws = [np.array(ff.w1.weight.astype(mx.float32)), np.array(ff.w3.weight.astype(mx.float32))]
        counts = _prefix_split([w.shape[0] for w in ws], share)
        gpus = [_suffix_module(w, a) for w, a in zip(ws, counts, strict=True)]
        self.w13 = WLFusedSplit(ws, gpus, counts, ws[0].shape[1], Mpad, compiled_ok)
        w2 = np.array(ff.w2.weight.astype(mx.float32))
        oa2 = w2.shape[0] - _round64(w2.shape[0] * share)
        self.w2 = WLSplitLinear(w2, _suffix_module(w2, oa2), oa2, w2.shape[1], Mpad, compiled_ok)

    def set_hack(self, v):
        self.w13.order_hack = v
        self.w2.order_hack = v

    def __call__(self, x):
        h1, h3 = self.w13(x)
        return self.w2(nn.silu(h1) * h3)
