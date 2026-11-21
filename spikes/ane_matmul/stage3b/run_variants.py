"""3b-1 overhead experiment driver: V0 concat vs E1/E2 partial-sum variants.

Correctness: each variant's eager block output vs V0, cosine target > 0.9999.
Timing: compiled+hack (production mode), interleaved rounds against
base_compiled and V0, one distinct callable per compiled mode (mx.compile
caches by function identity), macmon windows per mode.
Target from Scriv: block <= ~115 ms at T=4224.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import split_layers as SL  # noqa: E402
import variants as V  # noqa: E402
from ordering_block import build_transformer, capture_block_inputs, time_block  # noqa: E402

BLOCK = 15
ROUNDS = 4
CACHE = Path("/tmp/ane3a/cache3b")
RESULTS = HERE / "results"
MACMON = RESULTS / "variants_macmon.jsonl"


def cosine(a, b):
    an = mx.flatten(a.astype(mx.float32))
    bn = mx.flatten(b.astype(mx.float32))
    return float(mx.sum(an * bn) / (mx.sqrt(mx.sum(an * an)) * mx.sqrt(mx.sum(bn * bn))))


def main():
    t = build_transformer()
    inp = capture_block_inputs(t)
    blk = t.layers[BLOCK]
    orig_attn, orig_ff = blk.attention, blk.feed_forward
    args = (inp["x"], inp["attn_mask"], inp["freqs_cis"], inp["t_emb"])

    def set_pair(ap, fp):
        blk.attention = orig_attn if ap is None else ap
        blk.feed_forward = orig_ff if fp is None else fp

    def block_out(ap, fp):
        set_pair(ap, fp)
        y = blk(x=args[0], attn_mask=args[1], freqs_cis=args[2], t_emb=args[3])
        mx.eval(y)
        return y

    v0_a = SL.AttentionSplit(orig_attn, CACHE, BLOCK)
    v0_f = SL.FeedForwardSplit(orig_ff, CACHE, BLOCK)
    e2_a = V.AttentionHeadsSplit(orig_attn, CACHE, BLOCK)
    e1_f = V.FfRowsSplit(orig_ff, CACHE, BLOCK)

    pairs = {
        "base": (None, None),
        "V0": (v0_a, v0_f),
        "E2": (e2_a, v0_f),
        "E1": (v0_a, e1_f),
        "E12": (e2_a, e1_f),
    }

    y_v0 = block_out(v0_a, v0_f)
    RESULTS.mkdir(parents=True, exist_ok=True)
    corr = {"phase": "correctness"}
    for name in ("E2", "E1", "E12"):
        c = cosine(block_out(*pairs[name]), y_v0)
        corr[name] = round(c, 7)
        print(f"correctness {name}: cosine vs V0 = {c:.7f}", flush=True)
    with open(RESULTS / "variants.jsonl", "a") as f:
        f.write(json.dumps(corr) + "\n")

    # One distinct callable per compiled mode; force the trace in the right state.
    def make_fn():
        def fn(a_, m_, f_, te):
            return blk(x=a_, attn_mask=m_, freqs_cis=f_, t_emb=te)

        return fn

    compiled = {}
    for name, (ap, fp) in pairs.items():
        set_pair(ap, fp)
        compiled[name] = mx.compile(make_fn())
        mx.eval(compiled[name](*args))

    windows = {k: [] for k in pairs}
    best = {k: [] for k in pairs}
    mm = subprocess.Popen(["macmon", "pipe", "-i", "1000"], stdout=open(MACMON, "w"), stderr=subprocess.DEVNULL)
    try:
        for rnd in range(ROUNDS):
            for name in pairs:
                med, lo, hi, w0, w1 = time_block(lambda n=name: compiled[n](*args))
                best[name].append(med)
                windows[name].append((w0, w1))
            line = {"phase": "round", "round": rnd}
            line.update({k: round(best[k][-1], 1) for k in pairs})
            print(json.dumps(line), flush=True)
            with open(RESULTS / "variants.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
    finally:
        mm.terminate()
        set_pair(None, None)

    mrecs = []
    for l in open(MACMON):
        if l.strip():
            r = json.loads(l)
            r["_ts"] = datetime.fromisoformat(r["timestamp"]).timestamp()
            mrecs.append(r)

    def ane_activity(name):
        vals = []
        for w0, w1 in windows[name]:
            vals += [r["ane_power"] for r in mrecs if w0 <= r["_ts"] <= w1 and r.get("ane_power") is not None]
        return dict(mean=round(sum(vals) / len(vals), 2), mx=round(max(vals), 2)) if vals else None

    base = min(best["base"])
    summary = {"phase": "summary", "base_ms": round(base, 1)}
    for k in ("V0", "E2", "E1", "E12"):
        summary[k + "_ms"] = round(min(best[k]), 1)
        summary[k + "_saved_ms"] = round(base - summary[k + "_ms"], 1)
    summary["ane_activity"] = {k: ane_activity(k) for k in pairs}
    print(json.dumps(summary), flush=True)
    with open(RESULTS / "variants.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = "/Users/ianscrivener/MFlux/mflux-upstream/spikes/BENCH.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"owner": f"3b1-variants-{os.getpid()}", "pid": os.getpid(), "at": time.time()}).encode())
    try:
        main()
    finally:
        os.close(fd)
        try:
            os.unlink(lock)
        except FileNotFoundError:
            pass
