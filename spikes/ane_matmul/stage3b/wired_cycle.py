"""3b-2 dynamic-weights, part 2: wired memory while cycling 34 blocks x 4 sites.

Launches bench/surface_run with the four weightless programs from part 1 and
one shared IOSurface set per shape, cycles 34 synthetic layer weight blobs
through them, and samples vm_stat "Pages wired down" plus macmon ane_power
throughout. This is the Draw Things handoff model: programs cached by shape,
weights copied in per call. Gate: total wired under 3 GB, per-predict within
20% of baked.

Run (inside the part-1 lock window, after part 1 rows exist):
  .venv/bin/python spikes/ane_matmul/stage3b/wired_cycle.py [--layers 34] [--iters 5] [--blit]
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weightless_probe as W  # SITES, CACHE, M

SURFACE_RUN = HERE.parent / "bench" / "surface_run"
RESULTS = HERE / "results"
PAGE = 16384


def wired_pages():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"Pages wired down:\s+(\d+)", out)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=34)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--blit", action="store_true")
    args = ap.parse_args()

    models = []
    for site, N, K in W.SITES:
        p = W.CACHE / f"wl_{site}.mlpackage"
        if not p.exists():
            print(f"missing {p} - run part 1 first")
            return
        models += [str(p), str(N), str(K)]

    stop = threading.Event()
    wired_samples, ane_samples = [], []
    base_wired = wired_pages()

    def sampler():
        mac = subprocess.Popen(
            ["macmon", "pipe", "-i", "1000"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        try:
            while not stop.is_set():
                wired_samples.append(wired_pages())
                line = mac.stdout.readline()
                if line:
                    try:
                        p = json.loads(line).get("ane_power")
                        if p:
                            ane_samples.append(p)
                    except json.JSONDecodeError:
                        pass
                time.sleep(0.2)
        finally:
            mac.terminate()

    th = threading.Thread(target=sampler)
    th.start()
    cmd = (
        [str(SURFACE_RUN), str(W.M), "--layers", str(args.layers), "--iters", str(args.iters)]
        + (["--blit"] if args.blit else [])
        + models
    )
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    stop.set()
    th.join(timeout=10)

    events = []
    summary = None
    buf = proc.stdout
    dec = json.JSONDecoder()
    idx = 0
    while idx < len(buf):
        brace = buf.find("{", idx)
        if brace < 0:
            break
        try:
            d, end = dec.raw_decode(buf[brace:])
        except json.JSONDecodeError:
            break
        idx = brace + end
        if d.get("event") == "cold":
            events.append(d)
        elif "calls" in d:
            summary = d
    ws = [round((w - base_wired) * PAGE / 1e6) for w in wired_samples if wired_pages() and w >= 0]
    rec = {
        "mode": "blit" if args.blit else "memcpy",
        "layers": args.layers,
        "iters": args.iters,
        "wall_s": round(time.time() - t0, 1),
        "wired_start_mb": round(base_wired * PAGE / 1e6),
        "wired_delta_min_mb": min(ws) if ws else None,
        "wired_delta_med_mb": int(statistics.median(ws)) if ws else None,
        "wired_delta_max_mb": max(ws) if ws else None,
        "ane_peak_w": round(max(ane_samples), 1) if ane_samples else None,
        "cold_first_ms": [e.get("predict_ms") for e in events],
        "runner_rc": proc.returncode,
        "summary": summary,
        "stderr_tail": (proc.stderr or "")[-200:] if proc.returncode else "",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "wired_cycle.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
