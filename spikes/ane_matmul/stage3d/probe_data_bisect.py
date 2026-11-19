"""C: data bisect. Production graph, four data cells, one program per scale mode.

Probe A (synthetic normal weights, fp16 activation) reached cos 0.99954 for the full y arm.
scale_check (bf16-initialised module weights, bf16-quantised activation) reaches 0.9707 in weight
mode and 0.9998 in output mode. Same op sequence, different numbers, so the variable is the data.

Cells = {weights: synthetic-fp32 | real-bf16-init} x {activation: fp16 | bf16-rounded}.
Each cell is compared against exact fp32 computed from the tensors the ANE actually sees, so
activation rounding is never counted as ANE error.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # stage3d
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
from mlp_program import DIM, HIDDEN, MlpSpec, reference_fp32  # noqa: E402
from mlx.utils import tree_map  # noqa: E402
from rowsplit import BridgeBackend, prepare_block_weights  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock  # noqa: E402

M = 1024
rng = np.random.default_rng(0)

# real-block weights, exactly as the production harness builds them
mx.random.seed(0)
blk = ZImageTransformerBlock(DIM, 30)
blk.update(tree_map(lambda p: p.astype(mx.bfloat16), blk.parameters()))
mx.eval(blk.parameters())
ff = blk.feed_forward
w_real = [ff.w1.weight, ff.w3.weight, ff.w2.weight]
w_synth = [
    np.random.default_rng(7).normal(0, 0.016, (HIDDEN, DIM)).astype(np.float32),
    np.random.default_rng(8).normal(0, 0.016, (HIDDEN, DIM)).astype(np.float32),
    np.random.default_rng(9).normal(0, 0.016, (DIM, HIDDEN)).astype(np.float32),
]

x_n = rng.normal(0, 1.0, (1, M, DIM)).astype(np.float32)
x_fp16 = x_n.astype(np.float16)
x_bf16 = np.array(mx.array(x_n).astype(mx.bfloat16).astype(mx.float16))  # bf16-rounded, then widened

bb = BridgeBackend(HERE / "cache")


def to_np32(w):
    return w if isinstance(w, np.ndarray) else np.array(w.astype(mx.float32))


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), float(np.linalg.norm(a - b) / np.linalg.norm(b))


print(f"{'weights':>14} {'act':>7} {'mode':>7} {'cos':>10} {'relerr':>9} {'gain':>8} {'nonfin':>7} {'peak':>8}")
for wtag, ws in (("synthetic", w_synth), ("real-bf16", w_real)):
    w32 = [to_np32(w) for w in ws]
    for atag, xin in (("fp16", x_fp16), ("bf16", x_bf16)):
        # reference uses the SAME tensor the ANE receives, so activation rounding is not ANE error
        ref = reference_fp32(xin, w32[0], w32[1], w32[2])
        for mode in ("weight", "output"):
            spec = MlpSpec(mpad=M, scale_mode=mode)
            bw = prepare_block_weights(spec, ws[0], ws[1], ws[2])
            y = np.asarray(bb.run(spec, bw, xin), dtype=np.float32) * bw.gain
            c, r = cos(y, ref)
            g = float(
                (ref.astype(np.float64).ravel() @ y.astype(np.float64).ravel())
                / (y.astype(np.float64).ravel() @ y.astype(np.float64).ravel())
            )
            print(
                f"{wtag:>14} {atag:>7} {mode:>7} {c:>10.7f} {r:>9.5f} {g:>8.5f} "
                f"{int((~np.isfinite(y)).sum()):>7} {float(np.abs(y).max()):>8.4f}"
            )

print(
    f"\nweight scale ranges: synthetic w1 max|row| {np.abs(w_synth[0]).max(axis=1).min():.5f}..{np.abs(w_synth[0]).max(axis=1).max():.5f}"
    f" | real w1 {np.abs(to_np32(w_real[0])).max(axis=1).min():.5f}..{np.abs(to_np32(w_real[0])).max(axis=1).max():.5f}"
)
