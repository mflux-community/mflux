"""Stage 3c step ladder: patch N real blocks with the weightless bridge and time steps.

Same protocol as the 3b-2 battery, one change: the ANE side is the weightless
bridge (4 shared programs), so live MLModel count does not grow with N. Each
row records: step median/min, peak RSS, wired delta (vm_stat), swap start/end,
in-run canary, MEMGATE verdict. Compiled mode, order hack on, share 0.50.

./spikes/ane_matmul/watchdog.sh .venv/bin/python spikes/ane_matmul/stage3c/wl_step_battery.py --blocks 32 [--refiner]
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPIKE / "stage3a"))
sys.path.insert(0, str(SPIKE / "stage3b"))
sys.path.insert(0, str(SPIKE / "ane_linear"))
import wl_layers  # noqa: E402
from ordering_block import build_transformer  # noqa: E402
from s3a import BenchLock  # noqa: E402

RESULTS = HERE / "results"
PAGE = 16384


def _patch_block(blk, Mpad):
    """Patch one block and drop the columns the ANE now owns (production rule).

    Build the wrappers first: they read and quantize the originals; zeroing
    the originals before construction would feed them empty weights.
    """
    attn, ff = blk.attention, blk.feed_forward
    wa = wl_layers.WLAttentionSplit(attn, Mpad, compiled_ok=True)
    wf = wl_layers.WLFeedForwardSplit(ff, Mpad, compiled_ok=True)
    for name in ("to_q", "to_k", "to_v"):
        lin = getattr(attn, name)
        lin.weight = mx.zeros((0, lin.weight.shape[1]), mx.bfloat16)
    o = attn.to_out[0]
    o.weight = mx.zeros((0, o.weight.shape[1]), mx.bfloat16)
    for name in ("w1", "w2", "w3"):
        lin = getattr(ff, name)
        lin.weight = mx.zeros((0, lin.weight.shape[1]), mx.bfloat16)
    blk.attention, blk.feed_forward = wa, wf


def wired_pages():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"Pages wired down:\s+(\d+)", out)
    return int(m.group(1)) if m else -1


def mem_state():
    try:
        pct = int(
            subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout.split()[-1].rstrip("%")
        )
    except (ValueError, IndexError):
        pct = -1
    sw = float(
        subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
        .stdout.split("used =")[1]
        .split("M")[0]
    )
    return pct, sw


def canary(n=5):
    rng = np.random.default_rng(101)
    cx = mx.array(rng.standard_normal((4224, 3840)).astype(np.float32)).astype(mx.bfloat16)
    cw = mx.array(rng.standard_normal((20480, 3840)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(cx @ cw.T)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(cx @ cw.T)
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(sorted(ts)[n // 2], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--refiner", action="store_true")
    ap.add_argument("--pass", dest="passno", type=int, default=1)
    args = ap.parse_args()

    pct, sw0 = mem_state()
    if pct < 25 or sw0 > 2048:
        print(json.dumps(dict(error=f"MEMGATE free={pct}% swap={sw0:.0f}MB")))
        sys.exit(3)
    with BenchLock(owner=f"3c-ladder-{args.blocks}"):
        wired0 = wired_pages()
        t = build_transformer()
        # Shapes at 1024x1024: unified T = 4128 (4096 image + 32 cap); refiner sees 4096.
        rng = np.random.default_rng(5)
        lat = mx.array(rng.standard_normal((16, 1, 128, 128)).astype(np.float32)).astype(mx.bfloat16)
        cap = mx.array(rng.standard_normal((32, 2560)).astype(np.float32)).astype(mx.bfloat16)
        sg = mx.zeros((8,), mx.float32)
        n_models_before = wired_pages()
        patched = 0
        if args.blocks or args.refiner:
            if args.refiner:
                for idx, blk in enumerate(t.noise_refiner):
                    _patch_block(blk, 4096)
                    patched += 1
            for idx, blk in enumerate(t.layers[: args.blocks]):
                _patch_block(blk, 4352)
                patched += 1
        wired_patched = wired_pages()

        def fwd(ts_):
            return t(x=lat, timestep=ts_, sigmas=sg, cap_feats=cap)

        step = mx.compile(fwd)
        for i in range(2):
            mx.eval(step(mx.array(float(i), mx.float32)))
        warm = [mx.array(float(i), mx.float32) for i in range(3)]
        for w in warm:
            mx.eval(step(w))
        ts_ms = []
        for i in range(8):
            t0 = time.perf_counter()
            mx.eval(step(mx.array(float(i), mx.float32)))
            ts_ms.append((time.perf_counter() - t0) * 1e3)
        wired_end = wired_pages()
        _, sw1 = mem_state()
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS
        rec = dict(
            blocks=args.blocks,
            refiner=args.refiner,
            patched=patched,
            passno=args.passno,
            step_med_ms=round(statistics.median(ts_ms), 1),
            step_min_ms=round(min(ts_ms), 1),
            step_all=[round(x, 1) for x in ts_ms],
            canary_med_ms=canary(),
            peak_rss_mb=round(rss, 1),
            wired_delta_build_mb=round((n_models_before - wired0) * PAGE / 1e6),
            wired_delta_patch_mb=round((wired_patched - n_models_before) * PAGE / 1e6),
            wired_delta_steps_mb=round((wired_end - wired_patched) * PAGE / 1e6),
            wired_delta_total_mb=round((wired_end - wired0) * PAGE / 1e6),
            swap_start_mb=round(sw0),
            swap_end_mb=round(sw1),
        )
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "wl_step_battery.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
