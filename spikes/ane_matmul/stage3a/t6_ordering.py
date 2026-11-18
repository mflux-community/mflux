"""Test 6: stream ordering without MFLUX_ANE_ORDER_HACK.

Runs the Stage 2 split layer in-process (synthetic w2 weights, 4128 tokens) with
three launch strategies and measures per-call wall time:
  none  : ANE node and GPU matmul built independently (Stage 2 default off).
  hack  : data dependency on a zeroed ANE-input slice (current MFLUX_ANE_ORDER_HACK).
  async : mx.async_eval on the ANE result right after the node is built.
All strategies must return identical outputs. Headroom factor is 1/32 per the PRD.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import s3a
from s3a import BenchLock, emit

sys.path.insert(0, str(s3a.SPIKE / "ane_linear"))
import _ane_linear  # noqa: E402
import mlx.core as mx  # noqa: E402
from ane_split import build_ane_model  # noqa: E402

T_TOKENS = 4128
T_FIXED = 4224
O, I = 3840, 10240
PRESCALE = 1.0 / 32.0  # PRD: FeedForward.FLOAT16_HEADROOM = 32


def make_layer(strategy, cache_dir):
    rng = np.random.default_rng(5)
    w = (rng.normal(size=(O, I)) * 0.02).astype(np.float32)
    og = O // 2
    oa = O - og
    wg = mx.array(w[oa:]).astype(mx.bfloat16)
    gq, gs, gb = mx.quantize(wg, group_size=64, bits=8)
    mx.eval(gq, gs, gb)
    path = build_ane_model(w[:oa], cache_dir, f"t6_{strategy}", ksplit=4)
    ane = _ane_linear.AneModel(path, T_FIXED, I, oa)
    x = mx.array(rng.normal(size=(1, T_TOKENS, I)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(x)

    def call(x):
        xf = x.reshape(-1, I)
        T = xf.shape[0]
        xp = mx.pad(xf, [(0, T_FIXED - T), (0, 0)]) if T < T_FIXED else xf
        xa = (xp * PRESCALE).astype(mx.float16)
        ya = _ane_linear.ane_linear(xa, ane)
        if strategy == "async":
            mx.async_eval(ya)
        gx = xp
        if strategy == "hack":
            gx = xp + (xa[:1, :1] * 0).astype(xp.dtype)
        yg = mx.quantized_matmul(gx, gq, gs, gb, transpose=True, group_size=64, bits=8)
        scale = mx.concatenate([mx.full((oa,), 1.0 / PRESCALE), mx.ones((og,))])
        y = mx.concatenate([ya, yg], axis=-1).astype(mx.float32) * scale
        return y[:T].astype(x.dtype)

    return call, x


def bench(call, x, iters=30):
    y = call(x)
    mx.eval(y)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = call(x)
        mx.eval(y)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3, y


def main():
    cache = Path("/tmp/ane3a/cache")
    cache.mkdir(parents=True, exist_ok=True)
    layers = {}
    outs = {}
    for strategy in ("none", "hack", "async"):
        layers[strategy] = make_layer(strategy, cache)

    # Correctness first (no lock needed): all strategies must agree.
    for strategy, (call, x) in layers.items():
        y = call(x)
        mx.eval(y)
        outs[strategy] = np.array(y.astype(mx.float32))
    ref = outs["hack"]
    for strategy, y in outs.items():
        d = float(np.abs(y - ref).max())
        emit("t6_ordering", dict(phase="correctness", strategy=strategy, max_abs_diff_vs_hack=d))

    # GPU-only anchor (no lock: short, but keep inside the lock session below anyway).
    rng = np.random.default_rng(5)
    w_full = mx.array((rng.normal(size=(O, I)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    fq, fs, fb = mx.quantize(w_full, group_size=64, bits=8)
    x0 = mx.array(rng.normal(size=(1, T_TOKENS, I)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(fq, fs, fb, x0)

    with BenchLock(owner=f"stage3a-t6-{os.getpid()}"):
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            mx.eval(mx.quantized_matmul(x0.reshape(-1, I), fq, fs, fb, transpose=True, group_size=64, bits=8))
            ts.append(time.perf_counter() - t0)
        ts.sort()
        gpu_only = ts[len(ts) // 2] * 1e3
        emit("t6_ordering", dict(phase="timing", strategy="gpu_only_q8", median_ms=gpu_only))
        for strategy, (call, x) in layers.items():
            med, _ = bench(call, x)
            emit("t6_ordering", dict(phase="timing", strategy=strategy, median_ms=med, speedup_vs_gpu=gpu_only / med))


if __name__ == "__main__":
    main()
