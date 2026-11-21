"""Does the eager chain call the fused QK-norm+RoPE kernel, and does it change the output?

  python fused_accuracy.py save MODE OUT.npz      MODE = old | new (old sets MFLUX_ZIMAGE_DISABLE_FUSED_QKNORM_ROPE=1)
  python fused_accuracy.py compare OLD.npz NEW.npz

Chain: 2 noise_refiner + 6 main layers, real weights, S=4224. Arms: A (one compiled chain) and C (eager + threaded ANE MLP).
Reports the number of times the fused kernel was reached while tracing, then the final-output cosine and relative L2.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if sys.argv[1] == "compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for k in ("A", "C"):
        x, y = a[k].astype(np.float64).ravel(), b[k].astype(np.float64).ravel()
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        rel = float(np.linalg.norm(x - y) / np.linalg.norm(x))
        print(
            f"{k}: old vs new  cosine {cos:.8f}  rel_l2 {rel:.3e}  max_abs {np.abs(x - y).max():.4f}  identical {bool((x == y).all())}"
        )
    sys.exit(0)

mode, out = sys.argv[2], sys.argv[3]
if mode == "old":
    os.environ["MFLUX_ZIMAGE_DISABLE_FUSED_QKNORM_ROPE"] = "1"
import mlx.core as mx  # noqa: E402
from eager_bench import SplitBlock, load_real_blocks, make_inputs, real_ane_fns  # noqa: E402
from mlp_program import MlpSpec, plan_split  # noqa: E402

import mflux.models.z_image.model.z_image_transformer.attention as att  # noqa: E402

calls = {"n": 0}
_orig = att.fused_qk_norm_rope


def counted(*a, **k):
    r = _orig(*a, **k)
    calls["n"] += 1 if r is not None else 0
    return r


att.fused_qk_norm_rope = counted

refiners, main_blocks = load_real_blocks(2, 6)
x, mask, freqs, t = make_inputs(4224, 3840)
m = plan_split(4224, 0.5)[0]
blocks = refiners + main_blocks
chain = mx.compile(lambda x, mk, f, t: _run(x, mk, f, t))


def _run(x, mk, f, t):
    for b in blocks:
        x = b(x, mk, f, t)
    return x


ya = chain(x, mask, freqs, t)
mx.eval(ya)
n_a = calls["n"]
calls["n"] = 0
spec = MlpSpec(mpad=m, scale_mode="output", range_fix=True)
fns, gains, _ = real_ane_fns(main_blocks, spec, HERE / "cache")
pool = ThreadPoolExecutor(max_workers=1)
ph = {"pre_wait": [], "gpu_enqueue": [], "join_wait": []}
cr = [mx.compile(lambda x, mk, f, t, b=b: b(x, mk, f, t)) for b in refiners]
sp = [SplitBlock(b, fn, g, m, pool, ph) for b, fn, g in zip(main_blocks, fns, gains)]
y = x
for c in cr + sp:
    y = c(y, mask, freqs, t)
mx.eval(y)
n_c = calls["n"]
print(
    f"mode={mode}: fused kernel reached {n_a} times tracing A and {n_c} times tracing C (8 blocks x 1 trace each when fused)"
)
np.savez(out, A=np.array(ya.astype(mx.float32)), C=np.array(y.astype(mx.float32)))
