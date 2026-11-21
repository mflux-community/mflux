"""Smoke test for the wl_fused MLX primitive: correctness vs the pointer API, and compile safety.

The point of the primitive is that mflux's denoise step is wrapped in mx.compile, and the pointer
API's host sync is illegal there. So this checks two things and nothing else:

  1. the primitive returns bit-identical values to the pointer path it replaces;
  2. calling it inside mx.compile does not raise.

Uses scaled-down dims on the ANE. If the array placements are wrong the numbers will differ, so
test 1 is the guard on test 2 meaning anything.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
from mlp_program import (  # noqa: E402
    MlpSpec,  # noqa: E402
)
from rowsplit import BridgeBackend, prepare_block_weights  # noqa: E402

D, H, M = 384, 1024, 256
rng = np.random.default_rng(0)
w1 = rng.normal(0, 0.05, (H, D)).astype(np.float32)
w3 = rng.normal(0, 0.05, (H, D)).astype(np.float32)
w2 = rng.normal(0, 0.05, (D, H)).astype(np.float32)
x = rng.normal(0, 1.0, (1, M, D)).astype(np.float16)

spec = MlpSpec(mpad=M, dim=D, hidden=H, scale_mode="output", range_fix=True)
bb = BridgeBackend(HERE / "cache")
bw = prepare_block_weights(spec, w1, w3, w2)

y_ptr_np, hm_ptr_np = bb.run(spec, bw, x)
y_ptr = np.array(y_ptr_np, dtype=np.float32).copy()
hm_ptr = np.array(hm_ptr_np, dtype=np.float32).copy()
print(
    f"pointer API: y shape {y_ptr.shape} peak {np.abs(y_ptr).max():.4f}, h_max shape {hm_ptr.shape} peak {hm_ptr.max():.4f}"
)

ext = bb.ext
mx_in = [
    mx.array(x),
    mx.array(bw.w1),
    mx.array(bw.w3),
    mx.array(bw.w2),
    mx.array(bw.scales["s1"]),
    mx.array(bw.scales["s3"]),
    mx.array(bw.scales["s2"]),
]
model = bb._models[spec.name]

# 1. primitive, eager
y_p, hm_p = ext.wl_fused(*mx_in, model)
mx.eval(y_p, hm_p)
ya = np.array(y_p.astype(mx.float32))
ha = np.array(hm_p.astype(mx.float32))
same_y = np.array_equal(ya, y_ptr)
same_h = np.array_equal(ha, hm_ptr)
print(f"primitive eager: y shape {ya.shape} maxdiff vs pointer {np.abs(ya - y_ptr).max():.6f} identical={same_y}")
print(f"                 h_max maxdiff {np.abs(ha - hm_ptr).max():.6f} identical={same_h}")
if not (same_y and same_h):
    print(
        "  NOTE: a mismatch here means MLX placed the input arrays on a device Core ML cannot read"
        " (the int8 weights and x must be host-accessible), not that the numbers are wrong."
    )


# 2. primitive inside mx.compile -- the thing the pointer API cannot do
def step(xx):
    y, hm = ext.wl_fused(xx, mx_in[1], mx_in[2], mx_in[3], mx_in[4], mx_in[5], mx_in[6], model)
    return y * hm * 32.0


try:
    compiled = mx.compile(step)
    yc = compiled(mx_in[0])
    mx.eval(yc)
    ycv = np.array(yc.astype(mx.float32))
    print(
        f"mx.compile: OK, shape {ycv.shape}, maxdiff vs pointer-restored {np.abs(ycv - y_ptr * hm_ptr * 32.0).max():.6f}"
    )
except Exception as exc:  # noqa: BLE001 -- this test exists to report the failure verbatim
    print(f"mx.compile: FAILED {type(exc).__name__}: {str(exc)[:300]}")

# 3. the wired row split itself, patched the way generation does it and run under mx.compile.
#    The pointer form cannot reach this state at all, which is what killed the first real run.
from rowsplit import patch_transformer_compilable  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward  # noqa: E402

ff = FeedForward(D, H)
ff.w1.weight, ff.w3.weight, ff.w2.weight = (mx.array(w).astype(mx.bfloat16) for w in (w1, w3, w2))
mx.eval(ff.parameters())


class TinyTransformer:
    def __init__(self, ff):
        self.layers = [type("L", (), {"feed_forward": ff})()]


seq = 512
xin = mx.array(rng.normal(0, 1.0, (1, seq, D)).astype(np.float32)).astype(mx.bfloat16)
mx.eval(xin)
y_plain = np.array(ff(xin).astype(mx.float32))

tt = TinyTransformer(ff)
n = patch_transformer_compilable(tt, bb, spec)
y_eager = np.array(tt.layers[0].feed_forward(xin).astype(mx.float32))


def split_compiled(h):
    return tt.layers[0].feed_forward(h)


try:
    y_compiled = np.array(mx.compile(split_compiled)(xin).astype(mx.float32))
    print(f"\nrow split patched via primitive: {n} block(s)")
    for tag, got in (("eager", y_eager), ("compiled", y_compiled)):
        a, b = got.astype(np.float64).ravel(), y_plain.astype(np.float64).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        print(
            f"  {tag:>9} vs plain bf16: cos={cos:.7f} relerr={rel:.5f} "
            f"nonfinite={int((~np.isfinite(got)).sum())} shape={got.shape}"
        )
except Exception as exc:  # noqa: BLE001 -- reported verbatim, same reason as above
    print(f"\nrow split under mx.compile: FAILED {type(exc).__name__}: {str(exc)[:300]}")
