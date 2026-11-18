"""MLX GPU baselines: bf16 matmul and 8-bit quantized matmul. Prints JSON lines."""

import json
import sys
import time

import mlx.core as mx

SHAPES = [(3840, 3840), (10240, 3840), (3840, 10240)]  # (out, in)
TOKENS = [int(a) for a in sys.argv[1:]] or [1152, 4224, 5952]


def bench(f, n=50):
    for _ in range(10):
        mx.eval(f())
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        mx.eval(f())
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[n // 2], ts[int(n * 0.9)]


for T in TOKENS:
    for O, I in SHAPES:
        x = mx.random.normal((T, I)).astype(mx.bfloat16)
        w = mx.random.normal((O, I)).astype(mx.bfloat16) * 0.02
        wq, sc, bi = mx.quantize(w, group_size=64, bits=8)
        mx.eval(x, w, wq, sc, bi)
        for name, f in [
            ("bf16", lambda: x @ w.T),
            ("q8", lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=8)),
        ]:
            med, p90 = bench(f)
            print(
                json.dumps(
                    dict(
                        kind=name,
                        T=T,
                        O=O,
                        I=I,
                        median_ms=med * 1e3,
                        p90_ms=p90 * 1e3,
                        tflops=2 * T * O * I / med / 1e12,
                    )
                ),
                flush=True,
            )
