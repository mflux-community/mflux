"""Opt-in column split of one DiT linear layer: GPU (MLX 8-bit) + ANE (Core ML int8 W8A16).

Enable with env MFLUX_ANE_LINEAR=1 (see apply_ane_split). Default mflux behaviour is unchanged.
Both halves are quantised from the ORIGINAL bf16 weights.
"""

import hashlib
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import gen_models as G
from ckpt import load_bf16_weight  # noqa: F401  (re-exported)

_ext = None


def _extension():
    """Import the native extension on first use, so the flag-off path never needs it."""
    global _ext
    if _ext is None:
        import _ane_linear

        _ext = _ane_linear
    return _ext


_ORDER_HACK = os.environ.get("MFLUX_ANE_ORDER_HACK", "1") == "1"
T_FIXED = 4224  # 1024x1024: 4096 image tokens + up to 128 text tokens
PRESCALE = 1.0 / 16.0  # fp16 headroom: w2 outputs reach ~80k > 65504; undone on the GPU side


def enabled() -> bool:
    return os.environ.get("MFLUX_ANE_LINEAR", "0") == "1"


def build_ane_model(w: np.ndarray, cache_dir: Path, tag: str, ksplit: int) -> str:
    """int8 symmetric per-output-channel quantisation of w[O, I]; returns the .mlpackage path."""
    import coremltools as ct

    O, I = w.shape
    digest = hashlib.sha1(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest()[:12]
    path = cache_dir / f"{tag}_{O}x{I}_T{T_FIXED}_k{ksplit}_{digest}.mlpackage"
    if path.exists():
        return str(path)
    sc = (np.abs(w).max(1) / 127.0).astype(np.float16)
    wq = np.clip(np.round(w / sc.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
    prog = G.build_conv(T_FIXED, O, I, ksplit=ksplit, w_s=(wq, sc), to=True)
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=G.TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    return str(path)


class AneSplitLinear(nn.Module):
    """y = x @ W^T with output columns [0:oa) on the ANE (int8, W8A16) and [oa:O) on the GPU (8-bit).

    The ANE node is built first so MLX queues it before the GPU matmul; the two then overlap.
    """

    def __init__(self, w_bf16: np.ndarray, gpu_share: float, cache_dir: Path, tag: str, ksplit: int = 4):
        super().__init__()
        O, I = w_bf16.shape
        self.O, self.I = O, I
        og = min(max(int(round(O * gpu_share / 64)) * 64, 0), O)  # GPU columns; 8-bit group size 64
        self.oa = O - og
        if og > 0:
            wg = mx.array(w_bf16[self.oa :]).astype(mx.bfloat16)
            self.gq, self.gs, self.gb = mx.quantize(wg, group_size=64, bits=8)
            mx.eval(self.gq, self.gs, self.gb)
        self.has_gpu = og > 0
        self._scale = mx.concatenate([mx.full((self.oa,), 1.0 / PRESCALE), mx.ones((og,))])
        self.ane = None
        if self.oa > 0:
            path = build_ane_model(w_bf16[: self.oa], cache_dir, tag + "_head", ksplit)
            self.ane = _extension().AneModel(path, T_FIXED, I, self.oa)

    def last_error(self) -> str:
        """Error text of the latest failed ANE call, or an empty string. Call after mx.eval."""
        return self.ane.last_error() if self.ane is not None else ""

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, self.I)
        T = xf.shape[0]
        if self.ane is None:  # all GPU
            y = mx.quantized_matmul(xf, self.gq, self.gs, self.gb, transpose=True, group_size=64, bits=8)
            return y.reshape(*shape[:-1], self.O)
        if T > T_FIXED:
            raise ValueError(f"ane_linear supports at most {T_FIXED} tokens, got {T}")
        xp = mx.pad(xf, [(0, T_FIXED - T), (0, 0)]) if T < T_FIXED else xf
        xa = (xp * PRESCALE).astype(mx.float16)  # scale first (exact, power of two), then cast: no Inf from the cast
        ya = _extension().ane_linear(xa, self.ane)  # CPU stream, async
        if not self.has_gpu:
            y = ya.astype(mx.float32) * (1.0 / PRESCALE)
        else:
            gx = xp
            if _ORDER_HACK:  # make the GPU matmul come after the ANE launch in the tape
                gx = xp + (xa[:1, :1] * 0).astype(xp.dtype)
            yg = mx.quantized_matmul(gx, self.gq, self.gs, self.gb, transpose=True, group_size=64, bits=8)
            # One join op: the only consumer of the ANE result, so nothing on the GPU waits for it early.
            y = mx.concatenate([ya, yg], axis=-1).astype(mx.float32) * self._scale
        return y[:T].astype(x.dtype).reshape(*shape[:-1], self.O)


def apply_ane_split(model, block: int = 15, gpu_share: float = 0.5, cache_dir=None):
    """Replace feed_forward.w2 of one DiT block. No-op unless MFLUX_ANE_LINEAR=1."""
    if not enabled():
        return model
    cache = Path(cache_dir or os.environ.get("MFLUX_ANE_CACHE", HERE / "cache"))
    key = f"layers.{block}.feed_forward.w2.weight"
    split = AneSplitLinear(load_bf16_weight(key), gpu_share, cache, f"b{block}w2")
    model.transformer.layers[block].feed_forward.w2 = split
    return model
