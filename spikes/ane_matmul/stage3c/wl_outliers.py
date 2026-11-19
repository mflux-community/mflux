"""Stage 3c outlier-token overflow retest on the weightless form.

Stage 3a proved the baked W8A8 pre form overflow-safe on 300x outlier tokens;
the weightless form must prove it on its own path: constant-scale dequantise
inside the program, fp32 epilogue outside. Expectation: int8 codes are bounded
at +-127 regardless of activation magnitude, and the epilogue runs in fp32, so
no intermediate can overflow. Measure it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ane_linear"))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from wl_layers import WLSplitLinear, align256, quantize_prefix  # noqa: E402


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    rng = np.random.default_rng(8)
    T, O, I, oa = 512, 1920, 3840, 960
    w = (rng.standard_normal((O, I)) * 0.02).astype(np.float32)
    x = (rng.standard_normal((T, I)) * 0.05).astype(np.float32)
    x[7] *= 300.0
    x[91, :] *= 50.0
    x[91, 3] = 4.2e4
    suffix = nn.Linear(I, O - oa, bias=False)
    suffix.weight = mx.array(w[oa:]).astype(mx.bfloat16)
    layer = WLSplitLinear(w, suffix, oa, I, align256(T), compiled_ok=False)
    y = layer(mx.array(x).astype(mx.bfloat16))
    mx.eval(y)
    y_np = np.array(y.astype(mx.float32))
    ref = x.astype(np.float64) @ w.T.astype(np.float64)
    print(
        f"outlier cos {cos(y_np, ref):.7f} finite {bool(np.isfinite(y_np).all())} err {layer.model.last_error()!r:.60}"
    )

    # fp16 range probe: raw dot/K at the extreme row
    codes, swK = quantize_prefix(w[:oa])
    x32 = mx.array(x.astype(np.float32))
    tx = mx.maximum(mx.max(mx.abs(x32), axis=1) / 127.0, mx.array(1e-8))
    q = mx.clip(mx.round(x32 / tx[:, None]), -127, 127).astype(mx.int8)
    raw = (q.astype(mx.float32) @ codes.astype(mx.float32).T) / I
    mx.eval(raw)
    raw_np = np.array(raw)
    true_ane = x.astype(np.float64) @ w[:oa].T.astype(np.float64)
    print(
        f"raw max |y| in fp16 range: {float(np.abs(raw_np).max()):.1f} "
        f"(fp16 max 65504), epilogue cos vs true prefix {cos(raw_np * np.array(tx)[:, None] * np.array(swK)[None, :], true_ane):.7f}"
    )


if __name__ == "__main__":
    main()
