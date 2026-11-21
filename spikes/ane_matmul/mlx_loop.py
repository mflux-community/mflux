"""MLX bf16 matmul loop over a time window. Usage: mlx_loop.py T O I W0 W1 (epoch secs). Prints median ms."""

import json
import sys
import time

import mlx.core as mx

T, O, I = map(int, sys.argv[1:4])
w0, w1 = float(sys.argv[4]), float(sys.argv[5])
x = mx.random.normal((T, I)).astype(mx.bfloat16)
w = (mx.random.normal((O, I)) * 0.02).astype(mx.bfloat16)
mx.eval(x, w)
for _ in range(10):
    mx.eval(x @ w.T)
ts = []
while time.time() < w1:
    st = time.time()
    mx.eval(x @ w.T)
    d = time.time() - st
    if st >= w0:
        ts.append(d)
if not ts:
    sys.exit("measurement window missed: no samples collected (start earlier or lengthen the window)")
ts.sort()
print(json.dumps(dict(median_ms=ts[len(ts) // 2] * 1e3, n=len(ts))))
