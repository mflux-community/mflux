"""Does the real-weight failure reproduce in the numpy mirror?

If it does, the failure is the program's own range design, not an ANE quirk: in `output` mode the w2
matmul multiplies h by raw +-127 int8 codes, and h is never shrunk (INPUT_SHRINK applies to x only,
and x's shrink is compensated by the s1/s3 folds). Partial sums then approach the fp16 limit of
65504, which the synthetic probes never reached because their weights were std 0.016 instead of the
real 0.17.

Prints, per block and activation scale: the max abs value of each w2 chunk partial sum in fp32, the
mirror's fp16 nonfinite count, and mirror vs bf16 reference cosine.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlp_program import (  # noqa: E402
    DIM,
    HEADROOM,
    HIDDEN,
    INPUT_SHRINK,
    MlpSpec,
    program_scales,
    quantize_per_row,
    reference_fp16,
)
from safetensors import safe_open  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward  # noqa: E402

S = 4224
M = 2048
REPO = Path.home() / ".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo"
SNAP = next(REPO.glob("snapshots/*")).resolve()
WMAP = json.loads((SNAP / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
SHARDS: dict[str, object] = {}


def read(name: str) -> np.ndarray:
    shard = str(SNAP / "transformer" / WMAP[name])
    sf = SHARDS.get(shard)
    if sf is None:
        sf = safe_open(shard, framework="numpy")
        SHARDS[shard] = sf
    return sf.get_tensor(name)


mx.random.seed(0)
base = mx.random.normal((1, S, DIM), key=mx.random.key(0))
spec = MlpSpec(mpad=M, scale_mode="output")

print(
    f"{'blk':>4} {'g':>3} | {'max|a|':>9} {'max|h|':>9} {'max w2 chunk':>13} {'mirror inf':>11} "
    f"{'mirror cos':>11} {'fp16 cap':>9}"
)
for i in (0, 1, 5, 15, 29):
    w1 = read(f"layers.{i}.feed_forward.w1.weight")
    w3 = read(f"layers.{i}.feed_forward.w3.weight")
    w2 = read(f"layers.{i}.feed_forward.w2.weight")
    nw = read(f"layers.{i}.ffn_norm1.weight")
    ff = FeedForward(DIM, HIDDEN)
    ff.w1.weight = mx.array(w1).astype(mx.bfloat16)
    ff.w3.weight = mx.array(w3).astype(mx.bfloat16)
    ff.w2.weight = mx.array(w2).astype(mx.bfloat16)
    norm = nn.RMSNorm(DIM)
    norm.weight = mx.array(nw).astype(mx.bfloat16)
    mx.eval(ff.parameters(), norm.parameters())

    (q1, s1), (q3, s3), (q2, s2) = quantize_per_row(w1), quantize_per_row(w3), quantize_per_row(w2)
    sc = program_scales(spec, s1, s3, s2)

    for g in (1.0, 4.0):
        xin = norm(base) * g
        mx.eval(xin)
        ref = np.array(ff(xin).astype(mx.float32)).reshape(-1, DIM)
        xh = np.array(xin.astype(mx.float16))[:, :M]

        xq = xh.astype(np.float32) / INPUT_SHRINK
        a = (xq @ (q1.astype(np.float32) * 1.0).T).astype(np.float16) * sc["s1"]
        b = (xq @ (q3.astype(np.float32) * 1.0).T).astype(np.float16) * sc["s3"]
        h = (a / (1.0 + np.exp(-a.astype(np.float32))) * (b / HEADROOM)).astype(np.float16)
        step = HIDDEN // spec.k2_chunks
        chunk_peak = 0.0
        for k in range(spec.k2_chunks):
            part = (
                h[:, k * step : (k + 1) * step].astype(np.float32)
                @ q2[:, k * step : (k + 1) * step].astype(np.float32).T
            )
            chunk_peak = max(chunk_peak, float(np.abs(part.astype(np.float16)).max()))

        y, stats = reference_fp16(xh, {"w1": q1, "w3": q3, "w2": q2}, sc, spec)
        y_restore = np.asarray(y, dtype=np.float32) * 32.0
        cos = float(
            (ref.ravel().astype(np.float64) @ y_restore.ravel().astype(np.float64))
            / (np.linalg.norm(ref) * np.linalg.norm(y_restore))
        )
        print(
            f"{i:>4} {g:>3.0f} | {np.abs(a).max():>9.1f} {np.abs(h).max():>9.1f} {chunk_peak:>13.1f} "
            f"{int((~np.isfinite(y)).sum()):>11} {cos:>11.7f} {65504.0:>9.0f}"
        )
    del ff, ref

print("\nIf mirror cos tracks the ANE's 0.985-0.79 and chunk peaks sit near 65504, the fix is")
print("range design (shrink h before the w2 matmul and fold that into s2), not an ANE quirk.")
