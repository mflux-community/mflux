"""Stage 3c prep cost: time the quantise, transpose and epilogue chains in MLX.

Separates what the weightless path adds around the ANE call at production
shape (T=4128, I=3840, oa=5760, Mpad=4352): (a) token quantise alone,
(b) quantise + transpose into the program layout (the reshape-materialise
trick), (c) the epilogue (transpose back, fp32 scales, cast). All interleaved
against an unpatched bf16 matmul of the same FLOPs for reference.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
T, I, OA, MPAD = 4128, 3840, 5760, 4352


def timed(fn, iters=30):
    for _ in range(5):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 3)


def main():
    rng = np.random.default_rng(3)
    xp = mx.array(rng.standard_normal((MPAD, I)).astype(np.float32)).astype(mx.bfloat16)
    raw = mx.array(rng.standard_normal((OA, MPAD)).astype(np.float32)).astype(mx.float16)
    swK = mx.array((rng.random(OA) * 0.4 + 0.02).astype(np.float32))

    def quantise():
        x32 = xp.astype(mx.float32)
        tx = mx.maximum(mx.max(mx.abs(x32), axis=1) / 127.0, mx.array(1e-8))
        q = mx.clip(mx.round(x32 / tx[:, None]), -127, 127).astype(mx.int8)
        return q, tx

    def quantise_transpose():
        q, tx = quantise()
        xt = mx.reshape(mx.transpose(q, (1, 0)), (I * MPAD,))
        return mx.reshape(xt, (I, MPAD)), tx

    def epilogue():
        yy = mx.transpose(raw, (1, 0))[:T]
        tx = mx.maximum(mx.max(mx.abs(xp.astype(mx.float32)), axis=1) / 127.0, mx.array(1e-8))
        return (yy.astype(mx.float32) * tx[:T, None] * swK[None, :]).astype(mx.bfloat16)

    def reference_matmul():
        return xp[:T] @ mx.array(rng.standard_normal((OA, I)).astype(np.float32)).astype(mx.bfloat16).T

    rec = dict(
        shape=[T, I, OA, MPAD],
        quantise_ms=timed(quantise),
        quantise_transpose_ms=timed(quantise_transpose),
        epilogue_ms=timed(epilogue),
        reference_bf16_matmul_ms=timed(reference_matmul),
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "wl_prep_cost.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
