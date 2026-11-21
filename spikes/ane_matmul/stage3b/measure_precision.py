"""Interleaved W8A16 vs W8A8 benchmark at the production fused shapes.

BENCH.lock + watchdog + macmon ANE-power canary. Each round runs both arms on
the same bf16 input, timed from the shared input through the layer output
(including quantise/pad/cast work, which is part of each arm's real cost).
Paired analysis per round; ratios are within-session so machine drift
cancels. Arms:
  w8a16: x -> fp16 -> ext.ane_linear (zero-copy input wrap)
  w8a8:  x -> per-token quantise int8 + pad -> ext.ane_linear_q8
         (the binding transposes and materialises channels-first on the GPU)
Shapes: fused qkv, o, fused w1w3, w2 K-split 4. Tier 4608, T = 4224.
"""

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
sys.path.insert(0, str(SPIKE / "stage3a"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPIKE / "ane_linear"))
import _ane_linear as ext  # noqa: E402
import ane_models as AM  # noqa: E402
from s3a import BenchLock  # noqa: E402

CACHE = HERE / "models_precision"
RESULTS = HERE / "results"
T, TIER = 4224, 4608
ROUNDS = 7
CALLS = 15
SHAPES = [("qkv", 11520, 3840, 1), ("o", 3840, 3840, 1), ("w13", 20480, 3840, 1), ("w2", 3840, 10240, 4)]


def emit(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "precision_ab.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def build_models(rng):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag, O, I, ks in SHAPES:
        w = (rng.standard_normal((O, I)) * 0.02).astype(np.float32)
        wq, sc = AM.quantize_perchannel(w)
        p16 = CACHE / f"{tag}_w8a16.mlpackage"
        pq8 = CACHE / f"{tag}_q8pre.mlpackage"
        if not p16.exists():
            AM.convert(*AM.build_enum_w8a16(wq, sc, ksplit=ks, tiers=AM.TIERS), p16)
        if not pq8.exists():
            AM.convert(*AM.build_enum_q8pre(wq, sc, ksplit=ks, tiers=AM.TIERS), pq8)
        out[tag] = (ext.AneModel(str(p16), AM.TIERS[-1], I, O), ext.AneModel(str(pq8), AM.TIERS[-1], I, O), I, O)
    return out


def time_call(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def main():
    rng = np.random.default_rng(11)
    models = build_models(rng)
    xs = {
        tag: mx.array(rng.standard_normal((T, I)).astype(np.float32)).astype(mx.bfloat16)
        for tag, (_, _, I, _) in models.items()
    }

    def arm16(tag):
        m16, _, _, _ = models[tag]
        y = ext.ane_linear(mx.pad(xs[tag].astype(mx.float16), [(0, TIER - T), (0, 0)]), m16)
        mx.eval(y)

    def armq8(tag):
        _, mq8, _, _ = models[tag]
        x32 = xs[tag].astype(mx.float32)
        am = mx.max(mx.abs(x32), axis=1, keepdims=False)
        tok = mx.maximum(am / 127.0, mx.array(1e-6))
        q = mx.clip(mx.round(x32 / tok[:, None]), -127, 127).astype(mx.int8)
        y = ext.ane_linear_q8(mx.pad(q, [(0, TIER - T), (0, 0)]), mx.pad(tok, [(0, TIER - T)]), mq8)
        mx.eval(y)

    # Warm both graphs once per shape (compilation of the MLX ops, ANE plan).
    for tag in models:
        arm16(tag)
        armq8(tag)

    maclog = RESULTS / "precision_macmon.jsonl"
    maclog.unlink(missing_ok=True)
    macmon = subprocess.Popen(
        ["macmon", "pipe", "-i", "1000"], stdout=open(maclog, "w"), stderr=subprocess.DEVNULL, text=True
    )
    per = {tag: {"w8a16": [], "w8a8": []} for tag in models}
    pair = {tag: [] for tag in models}
    try:
        with BenchLock(owner="precision-ab"):
            emit(dict(phase="start", rounds=ROUNDS, calls=CALLS, tier=TIER, T=T, shapes=[s[0] for s in SHAPES]))
            for r in range(ROUNDS):
                med = {}
                for arm, fn in (("w8a16", arm16), ("w8a8", armq8)):
                    for tag in models:
                        ts = [time_call(lambda t=tag, f=fn: f(t)) for _ in range(CALLS)]
                        ts.sort()
                        med[(arm, tag)] = ts[len(ts) // 2]
                        per[tag][arm].append(med[(arm, tag)])
                for tag in models:
                    ratio = med[("w8a8", tag)] / med[("w8a16", tag)]
                    pair[tag].append(ratio)
                emit(dict(phase="round", round=r, **{f"{a}_{t}": round(v, 2) for (a, t), v in med.items()}))
    finally:
        macmon.terminate()
        macmon.wait(timeout=10)

    ane_w = []
    for line in maclog.read_text().splitlines():
        try:
            rec = json.loads(line)
            if rec.get("ane_power") is not None:
                ane_w.append(float(rec["ane_power"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    emit(
        dict(
            phase="summary",
            medians={t: {a: round(statistics.median(v), 2) for a, v in d.items()} for t, d in per.items()},
            ratio_median={t: round(statistics.median(v), 4) for t, v in pair.items()},
            ratio_spread={t: round(max(v) - min(v), 4) for t, v in pair.items()},
            ane_power_median_w=round(statistics.median(ane_w), 2) if ane_w else None,
            ane_power_min_w=round(min(ane_w), 2) if ane_w else None,
        )
    )


if __name__ == "__main__":
    main()
