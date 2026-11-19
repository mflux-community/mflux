"""3b-2 wired-reduction probe: one block, six configs, wired + block time.

Question: can fixed shapes (no enum) and pair-fusion (one model per pair of
call sites) bring per-block wired memory to <=250 MB at block time <=~120 ms?
The 3b-2 verdict measured the production 4-enumerated-model block at ~1000 MB
wired and 116-124 ms. Configs:

  W4E4    four enumerated models, W8A16        (production baseline)
  W4E1    four fixed-shape models,  W8A16
  W4E1q8  four fixed-shape models,  W8A8
  W2E1    pair-fused (2 models), fixed shape, W8A16
  W2E1q8  pair-fused, fixed shape,  W8A8
  best    the lowest-wired config

Wired is the headline: it is what the 36 GB machine cannot afford. Block time
is measured on the same split graph (real ZImageAttention + FeedForward, GPU
suffix matmuls, SDPA, silu), eager and compiled, interleaved against W4E4 so
drift cancels. One block only. MEMGATE refuses a hot machine. Foreground, under
BENCH.lock. Synthetic weights, seeded.

./spikes/ane_matmul/watchdog.sh .venv/bin/python \
  spikes/ane_matmul/stage3b/wired_probe.py [--config W2E1q8]
"""

import argparse
import json
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
sys.path.insert(0, str(SPIKE / "ane_linear"))
import _ane_linear as ext  # noqa: E402
import ane_models as AM  # noqa: E402
import wired_probe_models as WP  # noqa: E402
from s3a import BenchLock  # noqa: E402

CACHE = Path("/tmp/ane3a/cache_wired")
RESULTS = HERE / "results"
LOCK = "/Users/ianscrivener/MFlux/mflux-upstream/spikes/BENCH.lock"
MIN_FREE_PCT = 25
MAX_SWAP_MB = 2048.0
R = 4608  # tier rows
D = 3840  # model dim
H = 10240  # feed-forward inner
HEADS, HD = 30, 128
PAGE = 16384


def wired_mb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if "wired down" in line:
            return int(line.split(":")[1].strip().rstrip(".")) * PAGE / 1e6
    return -1.0


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
    return pct >= MIN_FREE_PCT and sw <= MAX_SWAP_MB, pct, sw


def mkweight(o, i, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((o, i)) * 0.02).astype(np.float32)


def handle(path, I, O, T=R):
    return ext.AneModel(str(path), T, I, O)


def build_w4(wqkv, wo, w13, w2, prec, enum, tag):
    """Four separate models. Returns list of (AneModel, I, O, name)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    specs = [("qkv", wqkv, 3840, 5760), ("o", wo, 3840, 1920), ("w13", w13, 3840, 10240), ("w2", w2, 10240, 1920)]
    out = []
    for name, w, I, O in specs:
        ks = 4 if name == "w2" else 1
        q, s = AM.quantize_perchannel(w)
        if enum:
            prog, ci = (
                AM.build_enum_w8a16(q, s, ksplit=ks, tiers=AM.TIERS)
                if prec == "w8a16"
                else AM.build_enum_q8pre(q, s, ksplit=ks, tiers=AM.TIERS)
            )
        else:
            prog, ci = WP.build_fixed(q, s, prec, ksplit=ks, rows=R)
        p = CACHE / f"{tag}_{name}_{prec}_{'e' if enum else 'f'}.mlpackage"
        if not p.exists():
            AM.convert(prog, ci, p)
        out.append((handle(p, I, O), I, O, name))
    return out


def build_w2(wqkv, wo, w13, w2, prec, tag):
    """Pair-fused: M_A (qkv+w13 one conv), M_B (o+w2 one model, two inputs)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    qq, sq = AM.quantize_perchannel(wqkv)
    qw, sw = AM.quantize_perchannel(w13)
    prog, ci = WP.build_pair_a(qq, sq, qw, sw, prec, rows=R)
    pa = CACHE / f"{tag}_pairA_{prec}.mlpackage"
    if not pa.exists():
        AM.convert(prog, ci, pa)  # single output "y" from build_fixed; plain ane_linear path
    a = handle(pa, 3840, 5760 + 10240)

    qo, so = AM.quantize_perchannel(wo)
    q2, s2 = AM.quantize_perchannel(w2)
    prog, ci = WP.build_pair_b(qo, so, q2, s2, prec, ksplit2=4, rows=R)
    pb = CACHE / f"{tag}_pairB_{prec}.mlpackage"
    if not pb.exists():
        WP.convert_multi(prog, ci, pb, 1)
    b = ext.make_multi_model(
        str(pb),
        R,
        ["aq", "hq"] if prec == "w8a8" else ["a", "h"],
        [3840, 10240],
        [1, 1] if prec == "w8a8" else [0, 0],
        ["atok", "htok"] if prec == "w8a8" else ["", ""],
        ["y"],
    )
    return [(a, 3840, 16000, "pairA"), (b, (3840, 10240), (1920, 1920), "pairB")]


def predict(h, prec):
    """One ANE call on a handle. Handles are (model, I, O, name); a pair-B handle
    carries tuple I/O to mark its two-input form. Zero inputs still exercise the
    full plan path (latency and wired do not depend on the values)."""
    m, I, O, _ = h
    if isinstance(I, tuple):  # pair B: two inputs a,b
        I1, I2 = I
        O1, O2 = O
        a = mx.zeros((R, I1), mx.float32)
        b = mx.zeros((R, I2), mx.float32)
        if prec == "w8a8":
            at = mx.maximum(mx.max(mx.abs(a), axis=1) / 127.0, mx.array(1e-6))
            ht = mx.maximum(mx.max(mx.abs(b), axis=1) / 127.0, mx.array(1e-6))
            aq = mx.clip(mx.round(a / at[:, None]), -127, 127).astype(mx.int8)
            hq = mx.clip(mx.round(b / ht[:, None]), -127, 127).astype(mx.int8)
            mx.eval(ext.ane_multi_q8([aq, hq], [at, ht], m, [O1 + O2])[0])
        else:
            mx.eval(ext.ane_multi_16([a.astype(mx.float16), b.astype(mx.float16)], m, [O1 + O2])[0])
        return
    x = mx.zeros((R, I), mx.float32)
    if prec == "w8a8":
        tok = mx.maximum(mx.max(mx.abs(x), axis=1) / 127.0, mx.array(1e-6))
        q = mx.clip(mx.round(x / tok[:, None]), -127, 127).astype(mx.int8)
        mx.eval(ext.ane_linear_q8(q, tok, m))
    else:
        mx.eval(ext.ane_linear(x.astype(mx.float16), m))


def get_handles(cfg, tag):
    kind, prec, enum = CONFIGS[cfg]
    if kind == "w4":
        return build_w4(QKV, O_, W13, W2, prec, enum, tag), prec
    return build_w2(QKV, O_, W13, W2, prec, tag), prec


def parts_for(c):
    """Yield (name, thunk) per model; the thunk builds/loads and returns a predict descriptor."""
    kind, prec, enum = CONFIGS[c]
    CACHE.mkdir(parents=True, exist_ok=True)
    if kind == "w4":
        specs = [
            ("qkv", QKV, 3840, 5760, 1),
            ("o", O_, 3840, 1920, 1),
            ("w13", W13, 3840, 10240, 1),
            ("w2", W2, 10240, 1920, 4),
        ]
        for name, w, I, O, ks in specs:

            def mk(name=name, w=w, I=I, O=O, ks=ks):
                q, s = AM.quantize_perchannel(w)
                if enum:
                    prog, ci = (
                        AM.build_enum_w8a16(q, s, ksplit=ks, tiers=AM.TIERS)
                        if prec == "w8a16"
                        else AM.build_enum_q8pre(q, s, ksplit=ks, tiers=AM.TIERS)
                    )
                else:
                    prog, ci = WP.build_fixed(q, s, prec, ksplit=ks, rows=R)
                p = CACHE / f"{c}_{name}_{prec}_{'e' if enum else 'f'}.mlpackage"
                if not p.exists():
                    AM.convert(prog, ci, p)
                return (handle(p, I, O), I, O, name)

            yield name, mk
    else:

        def mk_a():
            qq, sq = AM.quantize_perchannel(QKV)
            qw, sw = AM.quantize_perchannel(W13)
            prog, ci = WP.build_pair_a(qq, sq, qw, sw, prec, rows=R)
            pa = CACHE / f"{c}_pairA_{prec}.mlpackage"
            if not pa.exists():
                AM.convert(prog, ci, pa)
            return (handle(pa, 3840, 16000), 3840, 16000, "pairA")

        def mk_b():
            qo, so = AM.quantize_perchannel(O_)
            q2, s2 = AM.quantize_perchannel(W2)
            prog, ci = WP.build_pair_b(qo, so, q2, s2, prec, ksplit2=4, rows=R)
            pb = CACHE / f"{c}_pairB_{prec}.mlpackage"
            if not pb.exists():
                WP.convert_multi(prog, ci, pb, 1)
            m = ext.make_multi_model(
                str(pb),
                R,
                ["aq", "hq"] if prec == "w8a8" else ["a", "h"],
                [3840, 10240],
                [1, 1] if prec == "w8a8" else [0, 0],
                ["atok", "htok"] if prec == "w8a8" else ["", ""],
                ["y"],
            )
            return (m, (3840, 10240), (1920, 1920), "pairB")

        yield "pairA", mk_a
        yield "pairB", mk_b


def measure(cfgs):
    out = {}
    for c in cfgs:
        prec = CONFIGS[c][1]
        base = wired_mb()
        rows = []
        total_load = 0
        total_pred = 0
        last = base
        for name, mk in parts_for(c):
            h = mk()
            wl = round(wired_mb() - last)
            for _ in range(3):
                predict(h, prec)
            wp = round(wired_mb() - last - wl)
            rows.append(dict(model=name, load_mb=wl, predict_mb=wp))
            total_load += wl
            total_pred += wp
            last = wired_mb()
        out[c] = dict(
            prec=prec,
            models=len(rows),
            wired_load_mb=total_load,
            wired_predict_mb=total_pred,
            wired_total_mb=round(wired_mb() - base),
            per_model=rows,
        )
        emit(dict(phase="wired", config=c, **out[c]))
    return out


def measure_latency(cfgs, iters=15, rounds=6):
    hs = {c: get_handles(c, c) for c in cfgs}
    med = {c: [] for c in cfgs}
    for _ in range(rounds):
        for c in cfgs:
            handles, prec = hs[c]
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                for h in handles:
                    predict(h, prec)
                ts.append((time.perf_counter() - t0) * 1e3)
            med[c].append(sorted(ts)[iters // 2])
    return {c: round(sorted(med[c])[len(med[c]) // 2], 1) for c in cfgs}


def canary_med(n=5):
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


def emit(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "wired_probe.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


CONFIGS = {
    "W4E4": ("w4", "w8a16", True),
    "W4E1": ("w4", "w8a16", False),
    "W4E1q8": ("w4", "w8a8", False),
    "W2E1": ("w2", "w8a16", None),
    "W2E1q8": ("w2", "w8a8", None),
}


def main():
    global QKV, O_, W13, W2
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True, choices=["wired", "latency"])
    ap.add_argument("--config", default=None, help="single config for wired (fresh process: wired pages never return)")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma list for latency (interleaved, one process)")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    QKV, O_, W13, W2 = (
        mkweight(5760, 3840, 11),
        mkweight(1920, 3840, 12),
        mkweight(10240, 3840, 13),
        mkweight(1920, 10240, 14),
    )
    ok, pct, sw = memgate()
    emit(dict(phase="memgate", measure=args.measure, config=args.config, ok=ok, free_pct=pct, swap_mb=sw))
    if not ok:
        return
    with BenchLock(owner="wired-probe"):
        if args.measure == "wired":
            c = args.config or "W4E4"
            wired = measure([c])  # one pass per process: models stay loaded, so repeats would inflate
            emit(dict(phase="summary", measure="wired", config=c, **wired[c], canary_med_ms=canary_med()))
        else:
            cfgs = [c for c in args.configs.split(",") if c in CONFIGS]
            emit(dict(phase="start", configs=cfgs, canary_med_ms=canary_med()))
            for _ in range(2):
                lat = measure_latency(cfgs)
                emit(dict(phase="summary", measure="latency", ane_lat_ms=lat, canary_after_ms=canary_med()))


if __name__ == "__main__":
    main()
