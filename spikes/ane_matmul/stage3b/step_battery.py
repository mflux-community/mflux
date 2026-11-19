"""3b-2 whole-step battery: one arm in one process. --arm base|ane16|ane8 --res 1024|512.

Random bf16 ZImageTransformer, compiled single-step forward (the production
predict path), synthetic latents and cap features. Emits one JSON line.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(SPIKE / "stage3a"))

from mflux.models.z_image.ane import ZImageAne  # noqa: E402
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer  # noqa: E402

STEPS = 8
WARM = 3
CACHE = Path("/tmp/ane3a/step_cache")


def build_transformer():
    from mlx.utils import tree_map

    t = ZImageTransformer()
    t.update(
        tree_map(
            lambda v: v.astype(mx.bfloat16) if isinstance(v, mx.array) and v.dtype != mx.complex64 else v,
            t.parameters(),
        )
    )
    mx.eval(t.parameters())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "ane16", "ane8"])
    ap.add_argument("--res", type=int, required=True, choices=[512, 1024])
    ap.add_argument("--pass", dest="passno", type=int, default=0)
    args = ap.parse_args()

    t = build_transformer()
    patched = 0
    if args.arm != "base":
        t0 = time.perf_counter()
        patched = ZImageAne.apply(t, cache_dir=CACHE, precision="w8a8" if args.arm == "ane8" else "w8a16")
        patch_s = round(time.perf_counter() - t0, 1)
    else:
        patch_s = None

    side = args.res // 8
    rng = np.random.default_rng(5)
    lat = mx.array(rng.standard_normal((16, 1, side, side)).astype(np.float32)).astype(mx.bfloat16)
    cap = mx.array(rng.standard_normal((32, 2560)).astype(np.float32)).astype(mx.bfloat16)
    sg = mx.zeros((8,), mx.float32)

    if args.res == 512 and args.arm != "base":  # 27 s/step arms: keep the wall budget
        warm, nsteps = 2, 5
    else:
        warm, nsteps = WARM, STEPS
    step = mx.compile(lambda ts_: t(x=lat, timestep=ts_, sigmas=sg, cap_feats=cap))
    for i in range(warm):
        mx.eval(step(mx.array(float(i), mx.float32)))
    ts = []
    for i in range(nsteps):
        t0 = time.perf_counter()
        mx.eval(step(mx.array(float(i), mx.float32)))
        ts.append((time.perf_counter() - t0) * 1e3)
    # in-run GPU canary: fixed bf16 w1w3-shaped matmul, clean value ~57-62 ms
    cx = mx.array(rng.standard_normal((4224, 3840)).astype(np.float32)).astype(mx.bfloat16)
    cw = mx.array(rng.standard_normal((20480, 3840)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(cx @ cw.T)
    can = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(cx @ cw.T)
        can.append((time.perf_counter() - t0) * 1e3)
    print(
        json.dumps(
            dict(
                arm=args.arm,
                res=args.res,
                passno=args.passno,
                patched_blocks=patched,
                patch_s=patch_s,
                step_med_ms=round(statistics.median(ts), 1),
                step_min_ms=round(min(ts), 1),
                step_all=[round(x, 1) for x in ts],
                canary_med_ms=round(statistics.median(can), 1),
            )
        )
    )


if __name__ == "__main__":
    main()
