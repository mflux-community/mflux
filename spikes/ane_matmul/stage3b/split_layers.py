"""Stage 3b harness: split the four fused ANE call sites of one Z-Image block.

The 3a decisions inside the real transformer graph:
  - fused QKV (3840 -> 11520), one ANE call for the first oa columns,
  - o (3840^2), one ANE call,
  - fused w1w3 (3840 -> 20480), one ANE call,
  - w2 (10240 -> 3840), 4-chunk K-split, one ANE call, fp16 headroom 32.
GPU halves are bf16 matmuls of the retained columns. This harness measures the
overlap structure under mx.compile, not numerics, so it uses the random weights
of the instantiated transformer. No checkpoint is read.

order_hack: per-instance flag (the Stage 2 zero-dependency trick that launches
the ANE half first). Set it on every module before a run.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SPIKE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SPIKE / "ane_linear"))
import _ane_linear  # noqa: E402
from ane_split import build_ane_model  # noqa: E402

T_FIXED = 4224
W2_HEADROOM = 32.0  # FeedForward.FLOAT16_HEADROOM
ATTN_V_HEADROOM = 4.0  # ZImageAttention.FLOAT16_HEADROOM


def _round64(v):
    return max(64, int(round(v / 64)) * 64)


def _np32(a):
    """mx (possibly bf16) array -> np float32. numpy cannot view bf16 buffers."""
    return np.array(a.astype(mx.float32) if isinstance(a, mx.array) else a).astype(np.float32)


class _AneMixin:
    def _launch_ane(self, xp):
        """fp16 input for the ANE; prescale rows when headroom > 1."""
        xa = xp
        if self.headroom > 1.0:
            xa = (xp * (1.0 / self.headroom)).astype(mx.float16)
        else:
            xa = xp.astype(mx.float16)
        ya = _ane_linear.ane_linear(xa, self.ane)
        return xa, ya

    def _gpu_input(self, xp, xa):
        """The GPU-half input, with the ordering workaround when enabled."""
        if self.order_hack and self.ane is not None:
            return xp + (xa[:1, :1] * 0).astype(xp.dtype)
        return xp


class SplitLinear(_AneMixin, nn.Module):
    """One linear layer split column-wise: prefix oa on the ANE (int8), suffix on the GPU."""

    def __init__(self, w: np.ndarray, gpu_share: float, cache_dir: Path, tag: str, ksplit=1, headroom=1.0):
        super().__init__()
        O, I = w.shape
        og = min(max(_round64(O * gpu_share), 0), O)
        self.oa = O - og
        self.I = I
        self.headroom = headroom
        self.order_hack = True
        self.wg = mx.array(w[self.oa :]).astype(mx.bfloat16) if og > 0 else None
        self.ane = None
        if self.oa > 0:
            path = build_ane_model(w[: self.oa], cache_dir, tag, ksplit)
            self.ane = _ane_linear.AneModel(path, T_FIXED, I, self.oa)
        if headroom > 1.0:
            self._scale = mx.concatenate([mx.full((self.oa,), 1.0 * headroom), mx.ones((O - self.oa,))])
        else:
            self._scale = None

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, self.I)
        T = xf.shape[0]
        xp = mx.pad(xf, [(0, T_FIXED - T), (0, 0)]) if T < T_FIXED else xf
        if self.ane is None:
            y = mx.matmul(xp[:T], self.wg.T.astype(xp.dtype))
            return y.astype(x.dtype).reshape(*shp[:-1], self.wg.shape[0])
        xa, ya = self._launch_ane(xp)
        parts = [ya.astype(xp.dtype)]
        gx = self._gpu_input(xp, xa)
        if self.wg is not None:
            parts.append(mx.matmul(gx, self.wg.T.astype(xp.dtype)))
        y = mx.concatenate(parts, axis=-1)
        if self._scale is not None:
            y = y * self._scale.astype(y.dtype)
        return y[:T].astype(x.dtype).reshape(*shp[:-1], self.oa + (0 if self.wg is None else self.wg.shape[0]))


class FusedSplit(_AneMixin, nn.Module):
    """Several projections sharing one input; one ANE call over the fused output prefix."""

    def __init__(self, ws, gpu_share: float, cache_dir: Path, tag: str, headroom=1.0):
        super().__init__()
        Os = [w.shape[0] for w in ws]
        I = ws[0].shape[1]
        assert all(w.shape[1] == I for w in ws)
        O = sum(Os)
        og = min(max(_round64(O * gpu_share), 0), O)
        self.oa = O - og
        self.I = I
        self.headroom = headroom
        self.order_hack = True
        self.a_p = []
        self.gpu_w = []
        off = 0
        for d_p, w_p in zip(Os, ws):
            a = max(0, min(d_p, self.oa - off))
            self.a_p.append(a)
            self.gpu_w.append(mx.array(w_p[a:]).astype(mx.bfloat16) if d_p - a > 0 else None)
            off += d_p
        self.offs = np.cumsum([0] + Os[:-1]).tolist()
        self.ane = None
        if self.oa > 0:
            w_ane = np.concatenate([ws[i][: self.a_p[i]] for i in range(len(ws))], axis=0)
            path = build_ane_model(w_ane, cache_dir, tag, ksplit=1)
            self.ane = _ane_linear.AneModel(path, T_FIXED, I, self.oa)

    def __call__(self, x):
        """x [.., I] -> list of [.., D_p] outputs in ws order."""
        shp = x.shape
        xf = x.reshape(-1, self.I)
        T = xf.shape[0]
        xp = mx.pad(xf, [(0, T_FIXED - T), (0, 0)]) if T < T_FIXED else xf
        xa, ya = (None, None)
        if self.ane is not None:
            xa, ya = self._launch_ane(xp)
        outs = []
        for i, w in enumerate(self.gpu_w):
            parts = []
            a = self.a_p[i]
            if a > 0:
                parts.append(ya[:, self.offs[i] : self.offs[i] + a].astype(xp.dtype))
            if w is not None:
                gx = self._gpu_input(xp, xa)
                parts.append(mx.matmul(gx, w.T.astype(xp.dtype)))
            yf = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
            outs.append(yf[:T].astype(x.dtype).reshape(*shp[:-1], a + (0 if w is None else w.shape[0])))
        return outs


class AttentionSplit(nn.Module):
    """Replaces ZImageAttention for one block: fused QKV split + split output projection."""

    def __init__(self, attn: nn.Module, cache_dir: Path, block: int):
        super().__init__()
        ws = [_np32(attn.to_q.weight), _np32(attn.to_k.weight), _np32(attn.to_v.weight)]
        self.qkv = FusedSplit(ws, 0.5, cache_dir, f"b{block}_qkv")
        self.out = SplitLinear(_np32(attn.to_out[0].weight), 0.5, cache_dir, f"b{block}_o")
        self.a = attn

    def set_order_hack(self, v: bool):
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
    """Replaces FeedForward for one block: fused w1w3 split + split w2 (K-split, headroom 32)."""

    def __init__(self, ff: nn.Module, cache_dir: Path, block: int):
        super().__init__()
        ws = [_np32(ff.w1.weight), _np32(ff.w3.weight)]
        self.w13 = FusedSplit(ws, 0.5, cache_dir, f"b{block}_w13")
        self.w2 = SplitLinear(
            _np32(ff.w2.weight),
            0.5,
            cache_dir,
            f"b{block}_w2",
            ksplit=4,
            headroom=W2_HEADROOM,
        )

    def set_order_hack(self, v: bool):
        self.w13.order_hack = v
        self.w2.order_hack = v

    def __call__(self, x):
        h1, h3 = self.w13(x)
        hidden = nn.silu(h1) * (h3 * (1.0 / W2_HEADROOM))
        return self.w2(hidden)
