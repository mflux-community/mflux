"""GPU half of a split layer, windowed or fixed-iteration. Prints one JSON line.

Usage: gpu_half.py T O I MODE W0 W1
  MODE: bf16 | q8    W0 W1: epoch-second measurement window (pass 0 0 for 50 plain iterations)
"""

import json
import sys
import time

import mlx.core as mx

T, O, I = (int(a) for a in sys.argv[1:4])
mode = sys.argv[4]
w0, w1 = float(sys.argv[5]), float(sys.argv[6])

x = mx.random.normal((T, I)).astype(mx.bfloat16)
w = (mx.random.normal((O, I)) * 0.02).astype(mx.bfloat16)
if mode == "q8":
    wq, sc, bi = mx.quantize(w, group_size=64, bits=8)
    mx.eval(x, wq, sc, bi)

    def f():
        return mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=8)
elif mode == "bf16":
    mx.eval(x, w)

    def f():
        return x @ w.T
else:
    sys.exit(f"unknown mode {mode}")

for _ in range(10):
    mx.eval(f())
ts = []
if w1 > 0:
    while time.time() < w1:
        st = time.time()
        mx.eval(f())
        d = time.time() - st
        if st >= w0:
            ts.append(d)
    if not ts:
        sys.exit("measurement window missed: no samples collected (start earlier or lengthen the window)")
else:
    for _ in range(50):
        st = time.time()
        mx.eval(f())
        ts.append(time.time() - st)
ts.sort()
print(json.dumps(dict(mode=mode, T=T, O=O, I=I, median_ms=ts[len(ts) // 2] * 1e3, n=len(ts))))
