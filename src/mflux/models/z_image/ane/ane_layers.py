"""V0 split layers for the Z-Image DiT: column prefix on the ANE, suffix on the GPU.

One fused ANE call per call site (qkv, o, w1w3; w2 with K-split 4), the GPU
computing the remaining columns, one concat join per output. The measured
rules this implements (spikes/ane_matmul/stage3b, 3b-1/3b-2 reports):

- the shared padded input feeds BOTH halves and the ANE output is cast full
  width before any slicing, or the GPU suffix stops overlapping the ANE call;
- W8A8 pre form: per-token quantise in MLX, int8 in, the model applies the row
  scale after the matmul; the w2 headroom rides on that scale, not the grid;
- eager graphs drain the primitive input before the ANE call (DRAIN);
  mx.compile forbids the inner eval and schedules the chain itself.

Split layers assume the default stream precision (the gate refuses
--compute-precision and --float32 with --ane), so the stream is bfloat16 and
the only fp16 exposure is inside the ANE input cast, guarded by headroom.
"""

import mlx.core as mx
import mlx.nn as nn

from mflux.models.z_image.ane.ane_model_builder import AneModelBuilder


class AneHandle:
    """Launch wrapper for one native AneModel (the extension module is passed in)."""

    def __init__(self, ext, model):
        self._ext = ext
        self.model = model

    def linear(self, xa: mx.array) -> mx.array:
        return self._ext.ane_linear(xa, self.model)

    def linear_q8(self, q: mx.array, tok: mx.array) -> mx.array:
        return self._ext.ane_linear_q8(q, tok, self.model)


class _SplitBase(nn.Module):
    def __init__(self, prec: str, tiers, headroom: float):
        super().__init__()
        if prec not in ("w8a16", "w8a8"):
            raise ValueError(f"unknown ANE precision {prec}")
        self.prec = prec
        self.tiers = tiers
        self.headroom = headroom
        self.order_hack = True
        self.drain = True

    def _pad(self, xf: mx.array) -> tuple[mx.array, int]:
        T = xf.shape[0]
        tier = AneModelBuilder.pick_tier(T, self.tiers)
        return (mx.pad(xf, [(0, tier - T), (0, 0)]) if tier > T else xf), T

    def _launch_ane(self, xp: mx.array, xf: mx.array, handle: AneHandle) -> tuple[mx.array, mx.array]:
        """(ordering source, ANE output cast full width to xp.dtype)."""
        if self.prec == "w8a16":
            xa = (xp * (1.0 / self.headroom)).astype(mx.float16) if self.headroom > 1.0 else xp.astype(mx.float16)
            if self.drain:
                mx.eval(xa)
            return xa, handle.linear(xa).astype(xp.dtype)
        x32 = xf.astype(mx.float32)
        am = mx.max(mx.abs(x32), axis=1, keepdims=False)
        tok = mx.maximum(am / 127.0, mx.array(1e-6))
        q = mx.clip(mx.round(x32 / tok[:, None]), -127, 127).astype(mx.int8)
        # The model multiplies its output by tok, so the headroom undo rides on
        # tok: the quantise grid is unchanged and the fp16 model output stays
        # in range; _scale below restores it for both halves.
        tok = tok * (1.0 / self.headroom)
        pad = xp.shape[0] - xf.shape[0]
        if pad:
            q = mx.pad(q, [(0, pad), (0, 0)])
            tok = mx.pad(tok, [(0, pad)])
        if self.drain:
            mx.eval(q, tok)
        return q, handle.linear_q8(q, tok).astype(xp.dtype)

    def _gpu_input(self, xp: mx.array, xa: mx.array) -> mx.array:
        if self.order_hack:
            return xp + (xa[:1, :1] * 0).astype(xp.dtype)
        return xp


class SplitLinear(_SplitBase):
    """y = x @ W^T with columns [0:oa) on the ANE and [oa:O) on the GPU suffix."""

    def __init__(
        self, handle: AneHandle | None, gpu: nn.Module, oa: int, in_dim: int, prec: str, tiers, headroom: float = 1.0
    ):
        super().__init__(prec, tiers, headroom)
        self.handle = handle
        self.gpu = gpu
        self.oa = oa
        self.in_dim = in_dim
        self.O = oa + gpu.weight.shape[0]
        if headroom > 1.0:
            self._scale = mx.concatenate([mx.full((oa,), headroom), mx.ones((self.O - oa,))])
        else:
            self._scale = None

    def __call__(self, x: mx.array) -> mx.array:
        shp = x.shape
        xf = x.reshape(-1, self.in_dim)
        if self.handle is None:
            return self.gpu(xf).astype(x.dtype).reshape(*shp[:-1], self.O)
        xp, T = self._pad(xf)
        xa, y_full = self._launch_ane(xp, xf, self.handle)
        parts = [y_full[:T, : self.oa]]
        if self.O > self.oa:
            parts.append(self.gpu(self._gpu_input(xp, xa))[:T])
        y = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
        if self._scale is not None:
            y = y * self._scale.astype(y.dtype)
        return y.astype(x.dtype).reshape(*shp[:-1], self.O)


class FusedSplit(_SplitBase):
    """Projections sharing one input; one ANE call over the fused prefix.

    Returns a list of [.., d_p] outputs in projection order, each assembled
    from the fused ANE slice and the projection's own GPU suffix.
    """

    def __init__(
        self,
        handle: AneHandle | None,
        dims: list[int],
        ane_counts: list[int],
        gpus: list[nn.Module | None],
        in_dim: int,
        prec: str,
        tiers,
    ):
        super().__init__(prec, tiers, headroom=1.0)
        self.handle = handle
        self.dims = dims
        self.ane_counts = ane_counts
        self.gpus = gpus
        self.in_dim = in_dim
        self.offs = [0] * len(dims)
        off = 0
        for i, d in enumerate(dims):
            self.offs[i] = off
            off += d

    def __call__(self, x: mx.array) -> list[mx.array]:
        shp = x.shape
        xf = x.reshape(-1, self.in_dim)
        xp, T = self._pad(xf) if self.handle is not None else (xf, xf.shape[0])
        if self.handle is not None:
            xa, y_full = self._launch_ane(xp, xf, self.handle)
        outs = []
        for i, a in enumerate(self.ane_counts):
            gpu = self.gpus[i]
            parts = []
            if a > 0:
                parts.append(y_full[:T, self.offs[i] : self.offs[i] + a])
            if gpu is not None:
                gx = self._gpu_input(xp, xa) if self.handle is not None else xp
                parts.append(gpu(gx)[:T])
            y = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
            outs.append(y.astype(x.dtype).reshape(*shp[:-1], a + (0 if gpu is None else gpu.weight.shape[0])))
        return outs
