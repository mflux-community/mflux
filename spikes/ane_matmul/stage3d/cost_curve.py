"""Fixed vs marginal cost of one fused ANE MLP call, against the GPU on the same rows.

A row split wins only if max(ANE(m), GPU(S-m)) < GPU(S). This measures both curves at S=4224
so that gate can be computed before paying for the 34-block ladder.
One block's weights only -- no ladder, no real weights.
"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from mlp_program import DIM, HIDDEN, MlpSpec, plan_split  # noqa: E402
from rowsplit import BridgeBackend, prepare_block_weights  # noqa: E402

SEQ = 4224
REPS = 9
TIERS = [256, 512, 768, 1024, 1280, 2048, 4096]

rng = np.random.default_rng(0)
w1 = rng.normal(0, 0.02, (HIDDEN, DIM)).astype(np.float32)
w3 = rng.normal(0, 0.02, (HIDDEN, DIM)).astype(np.float32)
w2 = rng.normal(0, 0.02, (DIM, HIDDEN)).astype(np.float32)

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock  # noqa: E402

# The real block, so the GPU arm is the same module the ladder will run.
mx.random.seed(0)
blk = ZImageTransformerBlock(DIM, 30)
blk.update(tree_map(lambda p: p.astype(mx.bfloat16), blk.parameters()))
mx.eval(blk.parameters())
ff = blk.feed_forward


def gpu_ms(rows: int, reps: int = REPS) -> float:
    x = mx.random.normal((1, rows, DIM)).astype(mx.bfloat16)
    mx.eval(x)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y = ff(x)
        mx.eval(y)
        ts.append((time.perf_counter() - t0) * 1e3)
    del x, y
    return statistics.median(ts)


bb = BridgeBackend(HERE / "cache")
print(f"{'Mpad':>6}  {'ANE ms':>8}  {'GFLOP':>7}  {'TFLOPS':>7}  {'copybacks':>9}")
ane = {}
for mode in ("weight", "output"):
    print(f"\n--- scale_mode={mode} ---")
    for m in TIERS:
        spec = MlpSpec(mpad=m, scale_mode=mode)
        w = prepare_block_weights(spec, w1, w3, w2)
        x = np.ascontiguousarray(rng.normal(0, 1, (1, m, DIM)).astype(np.float16))
        bb.run(spec, w, x)  # warm: loads/compiles the program
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            y, _h_max = bb.run(spec, w, x)  # pre-range-fix arms, so h_max is None
            ts.append((time.perf_counter() - t0) * 1e3)
            float(y[0, 0, 0])
        med = statistics.median(ts)
        gf = 3 * m * HIDDEN * DIM * 2 / 1e9
        ane[(mode, m)] = med
        print(f"{m:>6}  {med:>8.2f}  {gf:>7.1f}  {gf / med:>7.2f}  {bb._models[spec.name].copybacks():>9}")
    del w, x

print(f"\n--- GPU bf16, same op, rows sweep (S={SEQ}) ---")
gpu = {}
for rows in [256, 512, 768, 1024, 1280, 2048, 3072, 4224]:
    gpu[rows] = gpu_ms(rows)
    print(f"{rows:>6} rows  {gpu[rows]:>8.2f} ms  {3 * rows * HIDDEN * DIM * 2 / 1e9 / gpu[rows]:>6.2f} TFLOPS")

full = gpu[SEQ]
print(f"\nGPU-only MLP for the whole block: {full:.2f} ms")
print(f"{'share':>6} {'m':>6} {'ANE':>7} {'GPU rest':>9} {'max(ideal)':>11} {'speedup':>8}  best-mode")
for share in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
    m, _ = plan_split(SEQ, share)
    if m not in TIERS:
        continue
    best = min(("weight", "output"), key=lambda mo: ane[(mo, m)])
    a = ane[(best, m)]
    g = gpu.get(SEQ - m)
    if g is None:
        continue
    ideal = max(a, g)
    print(f"{share:>6} {m:>6} {a:>7.2f} {g:>9.2f} {ideal:>11.2f} {full / ideal:>8.2f}x  {best}")

print("\nIdeal = perfect overlap, zero hand-off cost. Real T2 adds prep (np copy) + epilogue +")
print("concat on top of this, so every number above is the optimistic bound for that share.")
