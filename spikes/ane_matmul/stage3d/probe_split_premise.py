"""Is there anything to overlap? Time the GPU share at full vs split row counts.

The env matrix cut eval->task from 96 to 30 ms without moving the block period, and ops=4 had a
worse delay but a better ratio than ops=1. That says task-start latency is not the binding
constraint. This measures the other half of the premise: does handing 2048 of 4224 rows to the ANE
actually make the GPU's share cheaper, and what do the staging cast and fp32 epilogue cost?

One synthetic block at real Z-Image dims. No ANE call, no checkpoint, no image.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx  # noqa: E402
from bench import make_blocks  # noqa: E402
from mlp_program import DIM, plan_split  # noqa: E402

S = 4224
REPS = 20
M, _ = plan_split(S, 0.5)


def med(fn, *a):
    for _ in range(3):
        mx.eval(fn(*a))
    ts = []
    for _ in range(REPS):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(fn(*a))
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def main():
    ff = make_blocks(1)[0].feed_forward
    h = mx.random.normal((1, S, DIM)).astype(mx.bfloat16)
    mx.eval(h)
    h_gpu, h_ane = h[:, M:], h[:, :M]
    y16 = mx.random.normal((1, M, DIM)).astype(mx.float16)
    hm = mx.random.normal((1, M, 1)).astype(mx.float16)
    gain = mx.array(np.ones(DIM, dtype=np.float32))
    mx.eval(h_gpu, h_ane, y16, hm, gain)

    print(f"S={S} mpad={M} GPU rows={S - M}  ({REPS} reps, median ms)")
    rows = {}
    for name, fn, arg in [
        ("ff full 4224 rows", lambda a: ff(a), h),
        ("ff GPU share 2176 rows", lambda a: ff(a), h_gpu),
        ("ff ANE share 2048 rows", lambda a: ff(a), h_ane),
        ("staging cast 2048 rows", lambda a: a.astype(mx.float16), h_ane),
        ("epilogue fp32 restore", lambda a: a.astype(mx.float32) * hm.astype(mx.float32) * gain, y16),
    ]:
        eager = med(fn, arg)
        compiled = med(mx.compile(fn), arg)
        rows[name] = (eager, compiled)
        print(f"  {name:26s} eager {eager:7.2f}  compiled {compiled:7.2f}")

    full = rows["ff full 4224 rows"][1]
    share = rows["ff GPU share 2176 rows"][1]
    cast = rows["staging cast 2048 rows"][1]
    epi = rows["epilogue fp32 restore"][1]
    print(f"\nGPU share saves {full - share:.2f} ms of {full:.2f} ({(full - share) / full * 100:.1f}%)")
    print(f"staging + epilogue add {cast + epi:.2f} ms on the GPU stream")
    print(f"net GPU-side saving before any ANE overlap: {full - share - cast - epi:.2f} ms")
    print(
        f"ANE predict is ~52 ms, so perfect overlap would give {(full) / max(share + cast + epi, 52.0):.3f}x on the FFN alone"
    )


if __name__ == "__main__":
    main()
