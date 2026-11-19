"""Stage 3d real-weight gate: ANE output-mode split vs mflux GPU q8, per block, same reference.

Weights: Tongyi-MAI/Z-Image-Turbo transformer, read one block at a time from the F32 shards via the
index (the full transformer is 22.9 GB, and this gate needs three tensors per block).

Reference is the production path: bf16 mflux FeedForward (mflux casts the F32 checkpoint to bf16).

Arms per block, same input x:
  ane  -- rows [0, M) through the fused int8 W8A16 program (output-scaled), rows [M, S) on GPU bf16,
          rejoined on the token axis. This is the shipping shape.
  q8   -- the same FeedForward through mflux's own on-load quantisation (group 64, 8 bits).

Reported on two bases, because they answer different questions:
  full  -- error over all S rows. This is what a user would see, and it favours the split, since
           only its ANE rows are perturbed while q8 perturbs every row.
  ane_rows / q8_rows -- error over the rows each engine actually owns, for the like-for-like compare.

Activation: x = ffn_norm1(N(0,1)) * g, g in {1, 2, 4}, standing in for the (1 + scale_mlp)
modulation. Real modulation needs the text encoder and a denoising step, which this stage does not
run -- so treat the g sweep as a range, not a prediction.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # stage3d
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlp_program import DIM, HIDDEN, MlpSpec  # noqa: E402
from rowsplit import BridgeBackend, prepare_block_weights  # noqa: E402
from safetensors import safe_open  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward  # noqa: E402

S = 4224
SHARE = 0.50
M, _ = int(round(S * SHARE / 256)) * 256, 0
REPO = Path.home() / ".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo"
BLOCKS = list(range(30))
SEED = 0


def snapshot() -> Path:
    return next(REPO.glob("snapshots/*")).resolve()


SNAP = snapshot()
WEIGHT_MAP = json.loads((SNAP / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text())[
    "weight_map"
]
SHARD_CACHE: dict[str, object] = {}


def read_tensor(name: str) -> np.ndarray:
    shard = SNAP / "transformer" / WEIGHT_MAP[name]
    sf = SHARD_CACHE.get(str(shard))
    if sf is None:
        sf = safe_open(str(shard), framework="numpy")
        SHARD_CACHE[str(shard)] = sf
    return sf.get_tensor(name)


def metrics(y, ref):
    y = np.asarray(y, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    cos = float(r @ y / (np.linalg.norm(r) * np.linalg.norm(y)))
    rel = float(np.linalg.norm(y - r) / np.linalg.norm(r))
    gain = float(r @ y / (y @ y)) if y @ y else float("nan")
    return cos, rel, gain


bb = BridgeBackend(HERE / "cache")
spec = MlpSpec(mpad=M, scale_mode="output")
mx.random.seed(SEED)
base = mx.random.normal((1, S, DIM), key=mx.random.key(SEED))

print(f"real gate: Z-Image-Turbo, S={S}, share={SHARE} -> Mpad={M}, scale_mode=output, blocks={len(BLOCKS)}")
print(
    f"{'blk':>4} {'g':>3} | {'ANE full cos':>12} {'rel':>8} | {'q8 full cos':>12} {'rel':>8} | "
    f"{'ANErows':>8} {'q8rows':>8} | {'gate':>5} {'maxy':>8} {'nf':>3}"
)

rows = []
t0 = time.perf_counter()
for i in BLOCKS:
    w1 = read_tensor(f"layers.{i}.feed_forward.w1.weight")
    w3 = read_tensor(f"layers.{i}.feed_forward.w3.weight")
    w2 = read_tensor(f"layers.{i}.feed_forward.w2.weight")
    nw = read_tensor(f"layers.{i}.ffn_norm1.weight")

    ff = FeedForward(DIM, HIDDEN)
    ff.w1.weight = mx.array(w1).astype(mx.bfloat16)
    ff.w3.weight = mx.array(w3).astype(mx.bfloat16)
    ff.w2.weight = mx.array(w2).astype(mx.bfloat16)
    norm = nn.RMSNorm(DIM)
    norm.weight = mx.array(nw).astype(mx.bfloat16)
    mx.eval(ff.parameters(), norm.parameters())

    bw = prepare_block_weights(spec, w1, w3, w2)  # float32 numpy -> int8 codes + output scales
    x_gpu_rows = None
    for g in (1.0, 2.0, 4.0):
        xin = norm(base) * g
        mx.eval(xin)
        ref = ff(xin)
        mx.eval(ref)
        ref_np = np.array(ref.astype(mx.float32))

        x_np = np.array(xin.astype(mx.float16))
        y_ane = np.asarray(bb.run(spec, bw, x_np[:, :M]), dtype=np.float32) * bw.gain
        y_gpu = np.array(ff(xin[:, M:]).astype(mx.float32))
        y_split = np.concatenate([y_ane, y_gpu], axis=1)

        ffq = FeedForward(DIM, HIDDEN)
        ffq.w1 = ff.w1.to_quantized(group_size=64, bits=8)
        ffq.w3 = ff.w3.to_quantized(group_size=64, bits=8)
        ffq.w2 = ff.w2.to_quantized(group_size=64, bits=8)
        mx.eval(ffq.parameters())
        y_q8 = np.array(ffq(xin).astype(mx.float32))

        ac, ar, ag = metrics(y_split, ref_np)
        qc, qr, qg = metrics(y_q8, ref_np)
        ac_a, _, _ = metrics(y_ane, ref_np[:, :M])
        qc_a, _, _ = metrics(y_q8[:, :M], ref_np[:, :M])
        gate = "ANE" if ac >= qc else "q8"
        nf = int((~np.isfinite(y_split)).sum())
        print(
            f"{i:>4} {g:>3.0f} | {ac:>12.7f} {ar:>8.5f} | {qc:>12.7f} {qr:>8.5f} | "
            f"{ac_a:>8.5f} {qc_a:>8.5f} | {gate:>5} {float(np.abs(y_split).max()):>8.1f} {nf:>3}"
        )
        rows.append(
            {
                "block": i,
                "g": g,
                "ane_full_cos": ac,
                "ane_full_relerr": ar,
                "ane_gain": ag,
                "q8_full_cos": qc,
                "q8_full_relerr": qr,
                "q8_gain": qg,
                "ane_rows_cos": ac_a,
                "q8_rows_cos": qc_a,
                "nonfinite": nf,
                "max_abs_y": float(np.abs(y_split).max()),
                "w1_rowmax_spread": float(np.abs(w1).max(1).max() / np.abs(w1).max(1).min()),
                "w2_rowmax_spread": float(np.abs(w2).max(1).max() / np.abs(w2).max(1).min()),
            }
        )
    del ff, ffq, bw, x_np, y_ane, y_gpu, y_split, y_q8, ref_np

print(f"\nelapsed {time.perf_counter() - t0:.0f}s")
a = np.array([r["ane_full_cos"] for r in rows])
q = np.array([r["q8_full_cos"] for r in rows])
ar_ = np.array([r["ane_rows_cos"] for r in rows])
qr_ = np.array([r["q8_rows_cos"] for r in rows])
print(
    f"full-block cos: ANE min={a.min():.7f} median={np.median(a):.7f} | q8 min={q.min():.7f} median={np.median(q):.7f}"
)
print(
    f"owned-rows cos: ANE min={ar_.min():.7f} median={np.median(ar_):.7f} | q8 min={qr_.min():.7f} median={np.median(qr_):.7f}"
)
print(
    f"ANE >= q8 on full block in {int((a >= q).sum())}/{len(a)} rows; on owned rows in {int((ar_ >= qr_).sum())}/{len(ar_)}"
)
print(
    f"nonfinite total: {int(sum(r['nonfinite'] for r in rows))}, max |y| seen: {max(r['max_abs_y'] for r in rows):.1f}"
)
print(
    f"row-scale spread: w1 min={min(r['w1_rowmax_spread'] for r in rows):.2f} max={max(r['w1_rowmax_spread'] for r in rows):.2f}"
    f" | w2 min={min(r['w2_rowmax_spread'] for r in rows):.2f} max={max(r['w2_rowmax_spread'] for r in rows):.2f}"
)

out = HERE / "results" / f"real_gate_{int(time.time())}.json"
out.write_text(json.dumps(rows, indent=2))
print("wrote", out)
