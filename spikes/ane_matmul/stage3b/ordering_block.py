"""3b-0 (revised): block-level ordering gate under mx.compile.

The step-level battery could not see one patched block in a 5.2 s step (its
linear work is ~140 ms, 2.7% of the step, below noise). This driver times the
patched block in isolation, with inputs captured from one real forward.

Modes (interleaved every round, best round per mode):
  base_eager / base_compiled          unpatched block 15
  eager_hack / eager_nohack           split block, mx.compile off
  compiled_hack / compiled_nohack     split block, mx.compile on

Gate: saved_compiled = base_compiled - compiled_hack must be >= 0.9 x
saved_eager. Compiled nohack separates the compile effect from the ordering
effect: if compiled_nohack saves about as much as compiled_hack, compilation
kept overlap on its own; if it serialises like eager_nohack, the hack matters
under compile too.

Run under BENCH.lock + watchdog.sh in a foreground session after the throttle
probe. macmon samples ANE and GPU power for the whole session.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPIKE / "ane_linear"))
import split_layers as SL  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer  # noqa: E402

BLOCK = 15
ITERS = 20
ROUNDS = 4
CACHE = Path("/tmp/ane3a/cache3b")
RESULTS = HERE / "results"
MACMON = RESULTS / "ordering_block_macmon.jsonl"


def build_transformer():
    t = ZImageTransformer()
    t.update(
        tree_map(
            lambda v: v.astype(mx.bfloat16) if isinstance(v, mx.array) and v.dtype != mx.complex64 else v,
            t.parameters(),
        )
    )
    mx.eval(t.parameters())
    return t


def capture_block_inputs(t):
    """One real forward; keep block 15's four inputs via a proxy swap."""
    rng = np.random.default_rng(11)
    x = mx.array(rng.standard_normal((16, 1, 128, 128)).astype(np.float32)).astype(mx.bfloat16)
    cap = mx.array(rng.standard_normal((32, 2560)).astype(np.float32)).astype(mx.bfloat16)
    ts = mx.array(0.5, mx.float32)
    sg = mx.zeros((8,), mx.float32)
    blk = t.layers[BLOCK]

    class Proxy:
        def __init__(self):
            self.captured = None

        def __call__(self, *args, **kwargs):
            self.captured = kwargs
            return blk(*args, **kwargs)

    proxy = Proxy()
    t.layers[BLOCK] = proxy
    mx.eval(t(timestep=ts, x=x, cap_feats=cap, sigmas=sg))
    t.layers[BLOCK] = blk
    if proxy.captured is None:
        raise RuntimeError("capture failed: block was not called")
    return proxy.captured


def time_block(run, warm=3):
    for _ in range(warm):
        mx.eval(run())
    ts = []
    w0 = time.time()
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mx.eval(run())
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts), max(ts), w0, time.time()


def emit(record):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "ordering_block.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def main():
    t = build_transformer()
    inp = capture_block_inputs(t)
    blk = t.layers[BLOCK]
    orig_attn, orig_ff = blk.attention, blk.feed_forward
    args = [inp["x"], inp["attn_mask"], inp["freqs_cis"], inp["t_emb"]]
    print("captured:", {k: tuple(v.shape) if hasattr(v, "shape") else v for k, v in inp.items()})

    def block_call(u, m_, f_, te):
        return blk(x=u, attn_mask=m_, freqs_cis=f_, t_emb=te)

    # mx.compile caches graphs by function object and shapes. Every mode that
    # must trace a DIFFERENT graph needs its own callable, or a patched trace
    # silently reuses the earlier graph.
    base_compiled = mx.compile(block_call)
    set_state = lambda split: (  # noqa: E731
        setattr(blk, "attention", split[0] if split else orig_attn),
        setattr(blk, "feed_forward", split[1] if split else orig_ff),
    )
    mx.eval(base_compiled(*args))

    a_split = SL.AttentionSplit(orig_attn, CACHE, BLOCK)
    f_split = SL.FeedForwardSplit(orig_ff, CACHE, BLOCK)

    def set_hack(v):
        a_split.set_order_hack(v)
        f_split.set_order_hack(v)

    set_state((a_split, f_split))
    set_hack(True)
    comp_hack_fn = mx.compile(lambda u, m_, f_, te: blk(x=u, attn_mask=m_, freqs_cis=f_, t_emb=te))
    mx.eval(comp_hack_fn(*args))
    set_hack(False)
    comp_nohack_fn = mx.compile(lambda u, m_, f_, te: blk(x=u, attn_mask=m_, freqs_cis=f_, t_emb=te))
    mx.eval(comp_nohack_fn(*args))

    modes = {
        "base_eager": ("base", None),
        "base_compiled": ("base_c", None),
        "eager_hack": ("split_e", True),
        "eager_nohack": ("split_e", False),
        "compiled_hack": ("split_ch", None),
        "compiled_nohack": ("split_cn", None),
    }
    best = {k: [] for k in modes}
    windows = {k: [] for k in modes}
    mm = subprocess.Popen(["macmon", "pipe", "-i", "1000"], stdout=open(MACMON, "w"), stderr=subprocess.DEVNULL)
    try:
        for rnd in range(ROUNDS):
            for name, (kind, hack) in modes.items():
                if kind == "base" or kind == "base_c":
                    set_state(None)
                else:
                    set_state((a_split, f_split))
                    if hack is not None:
                        set_hack(hack)
                fn = {
                    "base": lambda: block_call(*args),
                    "base_c": lambda: base_compiled(*args),
                    "split_e": lambda: block_call(*args),
                    "split_ch": lambda: comp_hack_fn(*args),
                    "split_cn": lambda: comp_nohack_fn(*args),
                }[kind]
                med, lo, hi, w0, w1 = time_block(fn)
                best[name].append(med)
                windows[name].append((w0, w1))
            emit(dict(phase="round", round=rnd, **{k: round(v[-1], 1) for k, v in best.items()}))
    finally:
        mm.terminate()
        set_state(None)

    # ANE activity per mode, from macmon records inside each mode's timed windows.
    # This proves which graphs actually executed an ane_linear node.
    from datetime import datetime

    mrecs = []
    for l in open(MACMON):
        if not l.strip():
            continue
        r = json.loads(l)
        r["_ts"] = datetime.fromisoformat(r["timestamp"]).timestamp()
        mrecs.append(r)

    def ane_activity(name):
        vals = []
        for w0, w1 in windows[name]:
            vals += [r["ane_power"] for r in mrecs if w0 <= r["_ts"] <= w1 and r.get("ane_power") is not None]
        return dict(mean=round(sum(vals) / len(vals), 2), mx=round(max(vals), 2)) if vals else None

    emit(dict(phase="ane_activity", **{k: ane_activity(k) for k in modes}))

    base_c = min(best["base_compiled"])
    base_e = min(best["base_eager"])
    saved_eager = base_e - min(best["eager_hack"])
    saved_compiled = base_c - min(best["compiled_hack"])
    # Paired analysis: compute savings within each round, then take the median ratio.
    # Unpaired minima mix rounds measured minutes apart and distort the ratio.
    paired = []
    for i in range(len(best["base_eager"])):
        se = best["base_eager"][i] - best["eager_hack"][i]
        sc = best["base_compiled"][i] - best["compiled_hack"][i]
        if se > 0:
            paired.append(sc / se)
    med_ratio = statistics.median(paired) if paired else None
    ratio = saved_compiled / saved_eager if saved_eager > 0 else None
    verdict = "PASS" if med_ratio is not None and med_ratio >= 0.9 else "FAIL"
    emit(
        dict(
            phase="gate",
            base_eager_ms=round(base_e, 1),
            base_compiled_ms=round(base_c, 1),
            eager_hack_ms=round(min(best["eager_hack"]), 1),
            eager_nohack_ms=round(min(best["eager_nohack"]), 1),
            compiled_hack_ms=round(min(best["compiled_hack"]), 1),
            compiled_nohack_ms=round(min(best["compiled_nohack"]), 1),
            saved_eager_ms=round(saved_eager, 1),
            saved_compiled_ms=round(saved_compiled, 1),
            ratio=round(ratio, 3) if ratio else None,
            paired_ratios=[round(x, 3) for x in paired],
            median_paired_ratio=round(med_ratio, 3) if med_ratio else None,
            verdict=verdict,
        )
    )
    recs = [json.loads(l) for l in open(MACMON) if l.strip()]
    ap = sorted(r.get("ane_power", 0) for r in recs)
    gp = sorted(r.get("gpu_power", 0) for r in recs)
    emit(
        dict(
            phase="canary",
            n=len(recs),
            ane_power_med=ap[len(ap) // 2],
            ane_power_max=ap[-1],
            gpu_power_med=gp[len(gp) // 2],
        )
    )


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = "/Users/ianscrivener/MFlux/mflux-upstream/spikes/BENCH.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"owner": f"3b0-block-{os.getpid()}", "pid": os.getpid(), "at": time.time()}).encode())
    try:
        main()
    finally:
        os.close(fd)
        try:
            os.unlink(lock)
        except FileNotFoundError:
            pass
