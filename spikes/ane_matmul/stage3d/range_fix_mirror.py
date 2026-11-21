"""Range + accuracy search for the fused MLP, measured in the mirror only -- no Core ML, no ANE.

One explicit configuration per arm, so the magnitudes are checkable by eye rather than derived from
interacting flags:

  name     x shrink   matmul operands           in-program scales     h row-normalised
  cur      /128       raw +-127 codes           s*128 (w1,w3), s2     no    <- today, saturates
  k        none       codes / sqrt(K)           s*sqrt(K)             no    <- Draw Things constant
  hrow     /128       raw +-127 codes           s*128, s2             yes
  both     none       codes / sqrt(K)           s*sqrt(K)             yes   <- a + b
  g64h     none       int8 group-64 dequant     none (already folded) yes   <- q8-granularity weights

Gates, over all 34 FFN-bearing real blocks (30 layers + 2 noise_refiner + 2 context_refiner) and
g in {1,2,4}:
  range     max |float32 partial sum| <= 0.5 * 65504, and max |stored fp16 value| <= the same.
            h and the program output are stored in fp16 too, so they are inside the gate, not
            commentary on it.
  accuracy  cosine over rows [0,M) >= mflux q8 measured on the same block and activation.

The rowmax and the x32 headroom restore are applied in float32 AFTER the program's fp16 output,
because on the real path they live in the GPU epilogue. Applying them inside the fp16 chain was a
modelling bug in the first version of this script: it turned refiner rows into inf and made a
range-safe arm look broken.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlp_program import DIM, HEADROOM, HIDDEN, INPUT_SHRINK, quantize_per_row  # noqa: E402
from safetensors import safe_open  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward  # noqa: E402

S, M, SEED, GS = 4224, 2048, 0, (1.0, 2.0, 4.0)
K1, K2 = DIM, HIDDEN
SQ1, SQ2 = float(np.sqrt(K1)), float(np.sqrt(K2))
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


def groups_of(w: np.ndarray, groups: int):
    """Symmetric int8 with `groups` scales along K per row. mflux q8 at group_size=64 is groups=K/64."""
    out, k = w.shape
    wg = w.reshape(out, groups, k // groups)
    amax = np.maximum(np.abs(wg).max(axis=2), 1e-12)  # (out, groups); keepdims would break the repeat
    sc = amax / 127.0
    codes = np.clip(np.round(wg / sc[..., None]), -127, 127).reshape(out, k).astype(np.int8)
    return codes, sc


def peak(v) -> float:
    return float(mx.max(mx.abs(v)).item())


def chunked(x, w, k_chunks):
    """Partial sums in float32 (untruncated, for the range metric), accumulated as fp16 (stored)."""
    step = x.shape[-1] // k_chunks
    acc, hi = None, 0.0
    for i in range(k_chunks):
        lo, up = i * step, (i + 1) * step
        part = x[..., lo:up] @ w[:, lo:up].T
        hi = max(hi, peak(part))
        part = part.astype(mx.float16)
        acc = part if acc is None else (acc + part).astype(mx.float16)
    return acc, hi


def program(x, o1, o3, o2, p1, p3, p2, x_shrink, h_norm):
    xs = (x / INPUT_SHRINK).astype(mx.float16) if x_shrink else x
    pk: dict[str, float] = {}
    a, pk["w1"] = chunked(xs, o1, 1)
    a = (a.astype(mx.float32) * p1).astype(mx.float16) if p1 is not None else a
    b, pk["w3"] = chunked(xs, o3, 1)
    b = (b.astype(mx.float32) * p3).astype(mx.float16) if p3 is not None else b
    pk["a_store"], pk["b_store"] = peak(a), peak(b)
    b32 = (b.astype(mx.float32) / HEADROOM).astype(mx.float16)
    af = a.astype(mx.float32)
    h = ((af * mx.sigmoid(af)).astype(mx.float16) * b32).astype(mx.float16)
    pk["h_store"] = peak(h)
    if h_norm:
        rowmax = mx.maximum(mx.max(mx.abs(h.astype(mx.float32)), axis=-1, keepdims=True), 1e-8)
        hn = (h.astype(mx.float32) / rowmax).astype(mx.float16)
    else:
        hn, rowmax = h, None
    y, pk["w2"] = chunked(hn, o2, 4)
    y = (y.astype(mx.float32) * p2).astype(mx.float16) if p2 is not None else y
    pk["y_store"] = peak(y)
    return y, pk, rowmax


def cos_vs(y, ref):
    y, r = y.reshape(-1).astype(mx.float32), ref.reshape(-1).astype(mx.float32)
    return float((mx.sum(r * y)).item() / (mx.sqrt(mx.sum(y * y)).item() * mx.sqrt(mx.sum(r * r)).item()))


ARMS = ("cur", "k", "hrow", "both", "g64h")
mx.random.seed(SEED)
base = mx.random.normal((1, S, DIM), key=mx.random.key(SEED))

print(f"fp16 cap {CAP:.0f}  gate {LIMIT:.0f}  rows[0:{M}) of S={S}  blocks={len(BLOCKS)}  g={GS}")
print(f"{'block':>17} {'g':>2} | " + " | ".join(f"{a:>5} {'peak':>7} {'cos':>8}" for a in ARMS) + " |    q8 cos")
fails = {a: [] for a in ARMS}
worst = {a: 0.0 for a in ARMS}
t0 = time.perf_counter()

for blk in BLOCKS:
    ws = [read(f"{blk}.feed_forward.{n}.weight").astype(np.float32) for n in ("w1", "w3", "w2")]
    (q1, s1), (q3, s3), (q2, s2) = (quantize_per_row(w) for w in ws)
    R = dict(  # per-row codes, scale applied in-program on w1/w3, s2 on w2
        o1=mx.array(q1.astype(np.float32)),
        o3=mx.array(q3.astype(np.float32)),
        o2=mx.array(q2.astype(np.float32)),
        p1=mx.array(s1 * INPUT_SHRINK),
        p3=mx.array(s3 * INPUT_SHRINK),
        p2=mx.array(s2),
    )
    KK = dict(  # constant 1/sqrt(K): the operand is small, the real scale moves to the epilogue
        o1=R["o1"] / SQ1,
        o3=R["o3"] / SQ1,
        o2=R["o2"] / SQ2,
        p1=mx.array(s1 * SQ1),
        p3=mx.array(s3 * SQ1),
        p2=mx.array(s2 * SQ2),
    )
    G = {}  # mflux q8 granularity: group-64 scales, dequantised into the operand, nothing left post
    for key, w, k in (("o1", ws[0], K1), ("o3", ws[1], K1), ("o2", ws[2], K2)):
        codes, sc = groups_of(w, k // 64)
        G[key] = mx.array((codes.astype(np.float32) * np.repeat(sc, 64, axis=1).astype(np.float32)))

    ff = FeedForward(DIM, HIDDEN)
    ff.w1.weight, ff.w3.weight, ff.w2.weight = (mx.array(w).astype(mx.bfloat16) for w in ws)
    norm = nn.RMSNorm(DIM)
    norm.weight = mx.array(read(f"{blk}.ffn_norm1.weight")).astype(mx.bfloat16)
    mx.eval(ff.parameters(), norm.parameters())
    ffq = FeedForward(DIM, HIDDEN)
    ffq.w1, ffq.w3, ffq.w2 = (m.to_quantized(group_size=64, bits=8) for m in (ff.w1, ff.w3, ff.w2))
    mx.eval(ffq.parameters())

    for g in GS:
        xin = norm(base) * g
        mx.eval(xin)
        ref = ff(xin)[:, :M].astype(mx.float32)
        mx.eval(ref)
        x = xin[:, :M].astype(mx.float16).astype(mx.float32)
        q8cos = cos_vs(ffq(xin)[:, :M], ref)
        cells = []
        for name in ARMS:
            if name == "cur":
                y, pk, rm = program(x, R["o1"], R["o3"], R["o2"], R["p1"], R["p3"], R["p2"], True, False)
            elif name == "k":
                y, pk, rm = program(x, KK["o1"], KK["o3"], KK["o2"], KK["p1"], KK["p3"], KK["p2"], False, False)
            elif name == "hrow":
                y, pk, rm = program(x, R["o1"], R["o3"], R["o2"], R["p1"], R["p3"], R["p2"], True, True)
            elif name == "both":
                y, pk, rm = program(x, KK["o1"], KK["o3"], KK["o2"], KK["p1"], KK["p3"], KK["p2"], False, True)
            else:
                y, pk, rm = program(x, G["o1"], G["o3"], G["o2"], None, None, None, False, True)
            yfull = y.astype(mx.float32) * HEADROOM if rm is None else (y.astype(mx.float32) * rm * HEADROOM)
            rng = max(pk.values())
            worst[name] = max(worst[name], rng)
            c = cos_vs(yfull, ref)
            if not (rng <= LIMIT and c >= q8cos):
                fails[name].append((blk, g, round(rng, 1), round(c, 7), round(q8cos, 7)))
            cells.append(f"{rng:>7.0f} {c:>9.6f}")
        print(f"{blk:>17} {g:>2.0f} | " + " | ".join(cells) + f" | {q8cos:.6f}")
    del R, KK, G, ff, ffq

n = len(BLOCKS) * len(GS)
print(f"\n=== summary, {n} rows, {time.perf_counter() - t0:.0f}s ===")
print(f"{'arm':>6} {'worst gated value':>18} {'range ok':>11} {'range+acc ok':>13}")
for name in ARMS:
    bad = fails[name]
    rng_bad = sum(1 for x in bad if x[2] > LIMIT)
    print(
        f"{name:>6} {worst[name]:>18.1f} {n - rng_bad:>8}/{n} {n - len(bad):>10}/{n}"
        + ("" if not bad else f"   e.g. {bad[0]}")
    )
