"""Step 1d: T1 for the a+b range fix on real Z-Image-Turbo weights, through the bridge (the ANE).

All 34 FFN-bearing blocks (30 layers + 2 noise_refiner + 2 context_refiner), share 0.50 so the ANE
owns rows [0, 2048) of S=4224, activation g in {1,2,4} as in the real-weight gate.

Per block and g it reports:
  cos_ane   cosine of the restored ANE output vs the bf16 production reference, over the ANE's rows
  peak_y    max |program output| before the fp32 restore   (gate: <= 0.5 * 65504)
  peak_hmax max |h row max| handed to the epilogue         (gate: <= 0.5 * 65504)
  copybacks must stay 0, or the timings and the zero-copy claim are void

The range gate on partial sums is taken from range_fix_mirror.py, which already measured it for all
34 blocks x 3 g (arm "both": worst 5,060). Re-running the numpy mirror here would put ~100 CPU
matmuls of 241 GFLOP each on the critical path for a number that is already recorded.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlp_program import DIM, HIDDEN, MlpSpec, plan_split  # noqa: E402
from rowsplit import BridgeBackend, prepare_block_weights  # noqa: E402
from safetensors import safe_open  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward  # noqa: E402

S = 4224
SHARE = 0.50
GS = (1.0, 2.0, 4.0)
CAP, LIMIT = 65504.0, 0.5 * 65504
REPO = Path.home() / ".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo"
SNAP = next(REPO.glob("snapshots/*")).resolve()
WMAP = json.loads((SNAP / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
SHARDS: dict[str, object] = {}
BLOCKS = (
    [f"layers.{i}" for i in range(30)]
    + [f"noise_refiner.{i}" for i in range(2)]
    + [f"context_refiner.{i}" for i in range(2)]
)


def read(name: str) -> np.ndarray:
    shard = str(SNAP / "transformer" / WMAP[name])
    sf = SHARDS.get(shard)
    if sf is None:
        sf = safe_open(shard, framework="numpy")
        SHARDS[shard] = sf
    return sf.get_tensor(name)


M, _ = plan_split(S, SHARE)
spec = MlpSpec(mpad=M, scale_mode="output", range_fix=True)
bb = BridgeBackend(HERE / "cache")
mx.random.seed(0)
base = mx.random.normal((1, S, DIM), key=mx.random.key(0))

print(f"range-fix T1: spec={spec.name}  ANE rows=[0,{M}) of S={S}  blocks={len(BLOCKS)}  g={GS}")
print(f"{'block':>17} {'g':>2} | {'cos_ane':>10} {'peak_y':>9} {'peak_hmax':>10} {'nonfin':>6} | {'cos_full':>10}")
rows = []
t0 = time.perf_counter()
for blk in BLOCKS:
    ws = [read(f"{blk}.feed_forward.{n}.weight").astype(np.float32) for n in ("w1", "w3", "w2")]
    ff = FeedForward(DIM, HIDDEN)
    ff.w1.weight, ff.w3.weight, ff.w2.weight = (mx.array(w).astype(mx.bfloat16) for w in ws)
    norm = nn.RMSNorm(DIM)
    norm.weight = mx.array(read(f"{blk}.ffn_norm1.weight")).astype(mx.bfloat16)
    mx.eval(ff.parameters(), norm.parameters())
    bw = prepare_block_weights(spec, ws[0], ws[1], ws[2])

    for g in GS:
        xin = norm(base) * g
        mx.eval(xin)
        ref = np.array(ff(xin).astype(mx.float32))
        xh = np.array(xin.astype(mx.float16))
        y, hm = bb.run(spec, bw, xh[:, :M])
        y = np.array(y, dtype=np.float32)
        hm = np.array(hm, dtype=np.float32)
        y_res = y * hm * bw.gain  # fp32 epilogue restore, exactly as RowSplitFFN does it
        # Norms in float64: on float32 arrays of ~1e5 magnitude over 7.9M elements the accumulation
        # drifts enough to report a cosine above 1, which is not a physical value.
        ref_ane = ref[:, :M].astype(np.float64)
        y64 = y_res.astype(np.float64)
        cos_ane = float((ref_ane * y64).sum() / (np.linalg.norm(ref_ane) * np.linalg.norm(y64)))
        # full-block view: the GPU rows are exact by construction, so this is diluted by ~50%
        full = np.concatenate([y_res, np.array(ff(xin[:, M:]).astype(mx.float32))], axis=1)
        r64, f64 = ref.astype(np.float64), full.astype(np.float64)
        cos_full = float((r64 * f64).sum() / (np.linalg.norm(r64) * np.linalg.norm(f64)))
        peak_y, peak_hm = float(np.abs(y).max()), float(hm.max())
        nf = int((~np.isfinite(y_res)).sum())
        ok = peak_y <= LIMIT and peak_hm <= LIMIT and nf == 0
        rows.append(
            {
                "block": blk,
                "g": g,
                "cos_ane": cos_ane,
                "cos_full": cos_full,
                "peak_y": peak_y,
                "peak_hmax": peak_hm,
                "nonfinite": nf,
                "range_ok": ok,
            }
        )
        print(f"{blk:>17} {g:>2.0f} | {cos_ane:>10.7f} {peak_y:>9.2f} {peak_hm:>10.2f} {nf:>6} | {cos_full:>10.7f}")
    del ff, bw, ws

c = np.array([r["cos_ane"] for r in rows])
p = np.array([max(r["peak_y"], r["peak_hmax"]) for r in rows])
print(f"\n=== {len(rows)} rows, {time.perf_counter() - t0:.0f}s, copybacks={bb._models[spec.name].copybacks()} ===")
print(f"cos_ane  min={c.min():.7f} median={np.median(c):.7f}   (target ~0.99998 from the mirror)")
print(
    f"magnitudes  worst={p.max():.2f} vs limit {LIMIT:.0f}   range gate {int(sum(r['range_ok'] for r in rows))}/{len(rows)}"
)
print(f"rows below 0.9999: {int((c < 0.9999).sum())}   nonfinite total: {int(sum(r['nonfinite'] for r in rows))}")
out = HERE / "results" / f"real_t1_rf_{int(time.time())}.json"
out.write_text(json.dumps({"spec": spec.name, "rows": rows}, indent=2))
print("wrote", out)
