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
MIN_FREE_PCT = 25
MAX_SWAP_MB = 2048


def mem_state():
    """(free %, swap used MB) from the system tools; -1 when unreadable."""
    import re
    import subprocess

    try:
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, check=True).stdout
        free = int(re.search(r"free percentage: (\d+)%", out).group(1))
    except (OSError, AttributeError, ValueError):
        free = -1
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
        swap = float(re.search(r"used = ([\d.]+)M", sw).group(1))
    except (OSError, AttributeError, ValueError):
        swap = -1.0
    return free, swap


def peak_rss_mb():
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)  # bytes on macOS


def build_transformer():
    from mlx.utils import tree_map

    mx.random.seed(1234)  # identical weights across runs: the package cache keys on content
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
    ap.add_argument("--patch-first", type=int, default=None)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    t = build_transformer()
    patched = 0
    if args.arm != "base":
        t0 = time.perf_counter()
        patched = ZImageAne.apply(
            t,
            cache_dir=CACHE,
            precision="w8a8" if args.arm == "ane8" else "w8a16",
            limit=args.patch_first,
            refiner=args.patch_first is None,
        )
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
    free0, swap0 = mem_state()
    if free0 < MIN_FREE_PCT or swap0 > MAX_SWAP_MB:
        print(
            json.dumps(
                dict(arm=args.arm, res=args.res, passno=args.passno, error=f"MEMGATE free={free0}% swap={swap0:.0f}MB")
            )
        )
        sys.exit(3)

    def fwd(ts_):
        return t(x=lat, timestep=ts_, sigmas=sg, cap_feats=cap)

    step = fwd if args.eager else mx.compile(fwd)
    for i in range(warm):
        mx.eval(step(mx.array(float(i), mx.float32)))
        print(f"warm {i} ok", file=sys.stderr, flush=True)
    ts = []
    for i in range(nsteps):
        t0 = time.perf_counter()
        mx.eval(step(mx.array(float(i), mx.float32)))
        dt = (time.perf_counter() - t0) * 1e3
        ts.append(dt)
        print(f"step {i} {dt:.0f} ms", file=sys.stderr, flush=True)
    # in-run GPU canary: fixed bf16 w1w3-shaped matmul, clean value ~57-62 ms
    cx = mx.array(rng.standard_normal((4224, 3840)).astype(np.float32)).astype(mx.bfloat16)
    cw = mx.array(rng.standard_normal((20480, 3840)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(cx @ cw.T)
    can = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(cx @ cw.T)
        can.append((time.perf_counter() - t0) * 1e3)
    free1, swap1 = mem_state()
    print(
        json.dumps(
            dict(
                arm=args.arm,
                res=args.res,
                passno=args.passno,
                patched_blocks=patched,
                patch_s=patch_s,
                eager=args.eager,
                step_med_ms=round(statistics.median(ts), 1),
                step_min_ms=round(min(ts), 1),
                step_all=[round(x, 1) for x in ts],
                canary_med_ms=round(statistics.median(can), 1),
                peak_rss_mb=peak_rss_mb(),
                free_start_pct=free0,
                free_end_pct=free1,
                swap_start_mb=round(swap0),
                swap_end_mb=round(swap1),
            )
        )
    )


if __name__ == "__main__":
    main()
