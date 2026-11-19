"""Scale-sensitive check: cosine is invariant to a constant gain, so it cannot see a 128x error.

For each (scale_mode, w2_scale) config, compare the program output against the real bf16 FFN:
  ratio  = <y, ref> / <y, y>   (least-squares gain of ref on y; 1.0 means the scale is right)
  relerr = ||y - ref|| / ||ref||
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
from mlp_program import DIM, MlpSpec, reference_fp16  # noqa: E402
from mlx.utils import tree_map  # noqa: E402
from rowsplit import BridgeBackend, prepare_block_weights  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock  # noqa: E402

M = 1024
rng = np.random.default_rng(0)
mx.random.seed(0)

blk = ZImageTransformerBlock(DIM, 30)
blk.update(tree_map(lambda p: p.astype(mx.bfloat16), blk.parameters()))
mx.eval(blk.parameters())
ff = blk.feed_forward

x_mx = mx.random.normal((1, M, DIM), key=mx.random.key(0)).astype(mx.bfloat16)
mx.eval(x_mx)
ref = np.array(ff(x_mx).astype(mx.float32)).reshape(-1).astype(np.float64)
x_np = np.array(x_mx.astype(mx.float16))
print(f"reference FFN output: max|ref|={np.abs(ref).max():.4f}")

bb = BridgeBackend(HERE / "cache")

# Exact fp32 of the same FFN, as a second reference. T1 gates against the bf16 MLX module, so if
# the ANE diverges from one reference and not the other, the reference is part of the measurement.
w1f = np.array(ff.w1.weight.astype(mx.float32))
w3f = np.array(ff.w3.weight.astype(mx.float32))
w2f = np.array(ff.w2.weight.astype(mx.float32))
xf32 = np.array(x_mx.astype(mx.float32))
a32 = xf32 @ w1f.T
# yb carries the x32 gain (program returns y/32), so this must restore it too, or the fp32 column
# reads gain ~0.031 = 1/32 and looks like a 32x ANE error that is not there.
ref32 = (((a32 / (1.0 + np.exp(-a32))) * (xf32 @ w3f.T) / 32.0) @ w2f.T) * 32.0
print(f"exact fp32 FFN peak={np.abs(ref32).max():.4f}   bf16 module peak={np.abs(ref).max():.4f}")
ref32 = ref32.reshape(-1).astype(np.float64)

print(f"\n{'mode':>8} {'w2':>9} {'pair':>16} {'max|y|':>10} {'ratio':>9} {'relerr':>10} {'cos':>10}")
for scale_mode in ("weight", "output"):
    for w2_scale in ("program", "epilogue"):
        if w2_scale == "epilogue" and scale_mode == "weight":
            continue  # MlpSpec.validate forbids it: the w2 scale can only leave the program in output mode
        spec = MlpSpec(mpad=M, scale_mode=scale_mode, w2_scale=w2_scale)
        w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
        gain = w.gain.astype(np.float64)  # per-channel [D]: a mean would hide a scale bug
        yb = (np.asarray(bb.run(spec, w, x_np), dtype=np.float64).reshape(-1, DIM) * gain).reshape(-1)
        mirror, _ = reference_fp16(x_np, {"w1": w.w1, "w3": w.w3, "w2": w.w2}, w.scales, spec)
        ym = (mirror.astype(np.float64).reshape(-1, DIM) * gain).reshape(-1)
        for tag, y, other in (
            ("bridge vs bf16", yb, ref),
            ("bridge vs fp32", yb, ref32),
            ("mirror vs fp32", ym, ref32),
            ("bf16 vs fp32", ref, ref32),
        ):
            ratio = float(other @ y / (y @ y)) if y @ y else float("nan")
            relerr = float(np.linalg.norm(y - other) / np.linalg.norm(other))
            cos = float(other @ y / (np.linalg.norm(other) * np.linalg.norm(y)))
            print(
                f"{scale_mode:>8} {w2_scale:>9} {tag:>16} {np.abs(y).max():>10.4f} {ratio:>9.4f} {relerr:>10.5f} {cos:>10.6f}"
            )
