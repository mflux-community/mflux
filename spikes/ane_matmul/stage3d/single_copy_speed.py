"""Item 4 measurement: what does the GPU half cost if it reads the same 8-bit bytes as the ANE?

Today the FFN weights exist twice: bf16 for the GPU's rows and int8 for the ANE's rows. Deleting the
bf16 duplicate saves 236 MB per block (~7.1 GB over 30 main blocks), which is what should let the
ANE arm hold 1024^2 without swapping. The cost is that the GPU rows then run on 8-bit weights, and
Stage 3 measured q8 as slower than bf16 for Z-Image at production shapes -- so the memory fix could
give back the whole speedup. Measure that trade before wiring anything.

Arms, all at the real Z-Image FFN shapes, on the GPU-rows workload (S - Mpad rows):
  bf16       the current production path
  deq_call   dequantise our symmetric int8 codes to bf16 every call, then matmul (simplest sharing)
  qmm_g128   group-128 int8 through mx.quantized_matmul -- the coarsest granularity MLX allows
  qmm_g64    mflux's own q8 granularity, for reference

MLX rejects group sizes other than 32/64/128, so our per-row codes cannot feed the fused kernel at
all; that constraint is the reason deq_call is in this comparison.
"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlp_program import DIM, HIDDEN, quantize_per_row  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward  # noqa: E402

S_GPU = 2176  # 4224 - 2048: the rows the GPU keeps at share 0.50
REPS = 15
rng = np.random.default_rng(0)

# real-ish magnitudes: trained Z-Image FFN weights are std ~0.17, not nn.Linear init's 0.016
W = {n: rng.normal(0, 0.17, (HIDDEN, DIM)).astype(np.float32) for n in ("w1", "w3")}
W["w2"] = rng.normal(0, 0.17, (DIM, HIDDEN)).astype(np.float32)
x = mx.array(rng.standard_normal((1, S_GPU, DIM)).astype(np.float32)).astype(mx.bfloat16)
mx.eval(x)


def timeit(fn, reps=REPS):
    y = fn()
    mx.eval(y)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), y


def ff_with(w1, w3, w2):
    ff = FeedForward(DIM, HIDDEN)
    ff.w1.weight, ff.w3.weight, ff.w2.weight = w1, w3, w2
    mx.eval(ff.parameters())
    return ff


def dequantise(name):
    """Our symmetric per-row int8 codes widened to bf16 -- the same bytes the ANE program reads."""
    return (codes[name].astype(mx.float32) * scales[name][:, None]).astype(mx.bfloat16)


def qpair(name, group):
    q, s, b = mx.quantize(mx.array(W[name]), group_size=group, bits=8)
    return q, s, b


bf = {k: mx.array(v).astype(mx.bfloat16) for k, v in W.items()}
ff_bf16 = ff_with(bf["w1"], bf["w3"], bf["w2"])

codes, scales = {}, {}
for k, v in W.items():
    q, s = quantize_per_row(v)
    codes[k] = mx.array(q)
    scales[k] = mx.array(s.astype(np.float32))


def deq_per_call(h):
    a = mx.matmul(h, dequantise("w1").swapaxes(-1, -2))
    b = mx.matmul(h, dequantise("w3").swapaxes(-1, -2)) / 32.0
    hidden = nn.silu(a) * b
    return mx.matmul(hidden, dequantise("w2").swapaxes(-1, -2)) * 32.0


# MLX supports group sizes 32/64/128 only (verified: group_size=3840 raises
# "[quantize] The requested group size 3840 is not supported"), so a per-row shared 8-bit copy
# cannot use the fused quantized_matmul kernel. That is the constraint this measurement is for:
# the choices are dequantise-per-call at our granularity, or a finer group size shared by both
# engines. 128 is the coarsest supported, so it is the closest MLX can get to per-row.
g_128 = {k: qpair(k, 128) for k in ("w1", "w3", "w2")}
g_64 = {k: qpair(k, 64) for k in ("w1", "w3", "w2")}


def qmm(h, pack, group13, group2):
    q1, s1, b1 = pack["w1"]
    q3, s3, b3 = pack["w3"]
    q2, s2, b2 = pack["w2"]
    a = mx.quantized_matmul(h, q1, scales=s1, biases=b1, group_size=group13, bits=8)
    b = mx.quantized_matmul(h, q3, scales=s3, biases=b3, group_size=group13, bits=8) / 32.0
    hidden = nn.silu(a) * b
    return mx.quantized_matmul(hidden, q2, scales=s2, biases=b2, group_size=group2, bits=8) * 32.0


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


with mx.stream(mx.gpu):
    ref = np.array(ff_bf16(x).astype(mx.float32))
    t_bf, _ = timeit(lambda: ff_bf16(x))
    t_de, y_de = timeit(lambda: deq_per_call(x))
    t_12, y_12 = timeit(lambda: qmm(x, g_128, 128, 128))
    t_64, y_64 = timeit(lambda: qmm(x, g_64, 64, 64))

print(f"GPU rows: {S_GPU} of S=4224, real FFN shapes, std 0.17 weights, median of {REPS}")
print(f"{'arm':>12} {'ms':>8} {'vs bf16':>9} {'cos vs bf16':>12}")
for name, ms, y in (
    ("bf16", t_bf, None),
    ("deq_call", t_de, y_de),
    ("qmm_g128", t_12, y_12),
    ("qmm_g64", t_64, y_64),
):
    c = "-" if y is None else f"{cos(np.array(y.astype(mx.float32)), ref):.7f}"
    print(f"{name:>12} {ms:>8.2f} {t_bf / ms:>8.3f}x {c:>12}")

mb_bf16 = sum(v.itemsize * v.size for v in bf.values()) / 2**20
mb_int8 = sum(codes[k].astype(mx.uint8).itemsize * codes[k].size for k in codes) / 2**20
print(f"\nper block today: bf16 {mb_bf16:.0f} MB + int8 {mb_int8:.0f} MB = {mb_bf16 + mb_int8:.0f} MB")
print(
    f"one shared 8-bit copy: {mb_int8:.0f} MB -> saves {mb_bf16:.0f} MB/block, {mb_bf16 * 30 / 1024:.1f} GB over 30 main blocks"
)
