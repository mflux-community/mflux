"""Stage 3c block harness: weightless bridge vs baked, one real DiT block.

Correctness mode (runs freely): eager outputs of each arm against the
unpatched block and against the baked W8A8 arm. Gate: cosine >= 0.9999 vs
baked W8A8.
Timing mode (--timed, announce before running): interleaved rounds, compiled
+ order hack, MEMGATE + macmon + in-run canary; share sweep 0.50/0.45/0.40/
0.35/0.30. Gate: block <= ~115 ms.

Block 15 of a synthetic bf16 ZImageTransformer, inputs captured from one real
forward (T=4128 -> weightless Mpad 4352, baked tier 4608).

./spikes/ane_matmul/watchdog.sh .venv/bin/python spikes/ane_matmul/stage3c/wl_block_harness.py [--timed]
"""

import argparse
import json
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
import q8_layers  # noqa: E402
import wl_layers  # noqa: E402
from ordering_block import build_transformer, capture_block_inputs  # noqa: E402

BLOCK = 15
RESULTS = HERE / "results"
MACMON = RESULTS / "wl_block_macmon.jsonl"


def emit(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "wl_block_harness.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def canary(n=5):
    rng = np.random.default_rng(77)
    cx = mx.array(rng.standard_normal((4224, 3840)).astype(np.float32)).astype(mx.bfloat16)
    cw = mx.array(rng.standard_normal((20480, 3840)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(cx @ cw.T)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(cx @ cw.T)
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(sorted(ts)[n // 2], 1)


def arm_factory(kind, share, attn0, ff0, cache, Mpad, compiled):
    if kind == "base":
        return None
    if kind == "baked":
        prec = "w8a16" if share == "16" else "w8a8"
        a = q8_layers.AttentionSplit(attn0, cache, BLOCK, prec=prec, gpu_share=0.5)
        f = q8_layers.FeedForwardSplit(ff0, cache, BLOCK, prec=prec, gpu_share=0.5)
        for m in (a.qkv, a.out, f.w13, f.w2):
            m.DRAIN = not compiled
        return a, f
    a = wl_layers.WLAttentionSplit(attn0, Mpad, share=share, compiled_ok=compiled)
    f = wl_layers.WLFeedForwardSplit(ff0, Mpad, share=share, compiled_ok=compiled)
    return a, f


def run_correctness(blk, attn0, ff0, cache, Mpad, inp, args):
    def call():
        return blk(x=inp["x"], attn_mask=inp["attn_mask"], freqs_cis=inp["freqs_cis"], t_emb=inp["t_emb"])

    blk.attention, blk.feed_forward = attn0, ff0
    base = np.array(call()[0].astype(mx.float32))
    ref = None
    rows = []
    arms = [
        ("baked w8a8", arm_factory("baked", "8", attn0, ff0, cache, Mpad, False)),
        ("weightless 0.50", arm_factory("wl", 0.50, attn0, ff0, cache, Mpad, False)),
    ]
    arms += [
        (f"weightless {s:.2f}", arm_factory("wl", s, attn0, ff0, cache, Mpad, False)) for s in (0.45, 0.40, 0.35, 0.30)
    ]
    arms.append(("baked w8a16", arm_factory("baked", "16", attn0, ff0, cache, Mpad, False)))
    for name, split in arms:
        blk.attention, blk.feed_forward = (attn0, ff0) if split is None else split
        mx.eval(call())
        y = np.array(call()[0].astype(mx.float32))
        cos_base = cos(y, base)
        if "weightless" in name and split is not None:
            pass
        if name == "baked w8a8":
            ref = y
        cos_ref = cos(y, ref) if ref is not None and name != "baked w8a8" else None
        row = dict(arm=name, cos_vs_base=round(cos_base, 7), finite=bool(np.isfinite(y).all()))
        if cos_ref is not None:
            row["cos_vs_baked_w8a8"] = round(cos_ref, 7)
        rows.append(row)
        emit(dict(phase="cos", **row))
    blk.attention, blk.feed_forward = attn0, ff0
    gate = [r for r in rows if r["arm"].startswith("weightless")]
    ok = all(r.get("cos_vs_baked_w8a8", 1) >= 0.9999 and r["finite"] for r in gate)
    emit(dict(phase="cos_verdict", pass_=ok, n=len(gate)))
    return ok


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_timed(blk, attn0, ff0, cache, Mpad, inp, args):
    ok, pct, sw = memgate()
    emit(dict(phase="memgate", ok=ok, free_pct=pct, swap_mb=sw))
    if not ok:
        return
    args_ = [inp["x"], inp["attn_mask"], inp["freqs_cis"], inp["t_emb"]]
    specs = [("base", None, None), ("baked_w8a8", "baked", "8"), ("baked_w8a16", "baked", "16")]
    specs += [(f"wl_{s:.2f}", "wl", s) for s in (0.50, 0.45, 0.40, 0.35, 0.30)]
    fns = {}
    for name, kind, share in specs:
        split = arm_factory(kind, share, attn0, ff0, cache, Mpad, True) if kind else None
        blk.attention, blk.feed_forward = (attn0, ff0) if split is None else split
        fn = mx.compile(lambda u, m_, f_, te: blk(x=u, attn_mask=m_, freqs_cis=f_, t_emb=te))
        mx.eval(fn(*args_))
        fns[name] = fn
    macmon = subprocess.Popen(["macmon", "pipe", "-i", "1000"], stdout=open(MACMON, "w"), stderr=subprocess.DEVNULL)
    med = {n: [] for n, _, _ in specs}
    try:
        for rnd in range(args.rounds):
            for name, _, _ in specs:
                fn = fns[name]
                ts = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    mx.eval(fn(*args_))
                    ts.append((time.perf_counter() - t0) * 1e3)
                med[name].append(statistics.median(ts))
            emit(dict(phase="round", round=rnd, canary_ms=canary(), **{n: round(v[-1], 1) for n, v in med.items()}))
    finally:
        macmon.terminate()
        macmon.wait(timeout=10)
    summary = {n: round(statistics.median(v), 1) for n, v in med.items()}
    emit(
        dict(
            phase="timed_verdict",
            med_ms=summary,
            wl_best=min(s for n, s in summary.items() if n.startswith("wl")),
            gate_ms=115.0,
        )
    )


def memgate():
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
    return pct >= 25 and sw <= 2048.0, pct, sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timed", action="store_true")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()
    if args.timed:
        ok, _, _ = memgate()
        if not ok:
            print("run unannounced timing is refused: announce first")
            return
    t = build_transformer()
    inp = capture_block_inputs(t)
    blk = t.layers[BLOCK]
    attn0, ff0 = blk.attention, blk.feed_forward
    T = inp["x"].shape[1]
    Mpad = wl_layers.align256(T)
    cache = Path("/tmp/ane3a/cache_3c")
    cache.mkdir(parents=True, exist_ok=True)
    emit(dict(phase="start", T=T, Mpad=Mpad, timed=args.timed))
    run_correctness(blk, attn0, ff0, cache, Mpad, inp, args)
    if args.timed:
        run_timed(blk, attn0, ff0, cache, Mpad, inp, args)


if __name__ == "__main__":
    main()
