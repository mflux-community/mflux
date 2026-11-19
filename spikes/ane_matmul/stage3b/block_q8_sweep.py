"""3b-2 checkpoint: block-level W8A8 measurement and split-ratio re-sweep.

Block 15 of a synthetic bf16 ZImageTransformer, inputs captured at T=4224
(one real forward), arms interleaved every round, compiled + order-hack mode
(the production default). Arms:
  base          unpatched block
  w8a16@50      V0 split, W8A16, GPU:ANE 50:50
  w8a8@50/45/40 W8A8 pre path, GPU share 50/45/40 percent
  w8a16@45/40   context arms for the ratio sweep
Canaries: a fixed bf16 w1w3-shaped matmul timed every round (GPU contention),
macmon ANE power windows per arm. BENCH.lock + watchdog, foreground session.

Run: ./spikes/ane_matmul/watchdog.sh .venv/bin/python \
       spikes/ane_matmul/stage3b/block_q8_sweep.py
"""

import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPIKE / "stage3a"))
import q8_layers as QL  # noqa: E402
from ordering_block import build_transformer, capture_block_inputs  # noqa: E402

BLOCK = 15
ITERS = 8
ROUNDS = 6
CACHE = Path("/tmp/ane3a/cache_bq8")
RESULTS = HERE / "results"
MACMON = RESULTS / "block_q8_sweep_macmon.jsonl"
LOCK = "/Users/ianscrivener/MFlux/mflux-upstream/spikes/BENCH.lock"

ARMS = [
    ("base", None, None),
    ("w8a16@50", "w8a16", 0.50),
    ("w8a8@50", "w8a8", 0.50),
    ("w8a8@45", "w8a8", 0.45),
    ("w8a8@40", "w8a8", 0.40),
    ("w8a16@45", "w8a16", 0.45),
    ("w8a16@40", "w8a16", 0.40),
]

# fixed GPU canary op (bf16 matmul of the w1w3 fused shape)
CAN_X = mx.array(np.random.default_rng(3).standard_normal((4224, 3840)).astype(np.float32)).astype(mx.bfloat16)
CAN_W = mx.array(np.random.default_rng(4).standard_normal((20480, 3840)).astype(np.float32)).astype(mx.bfloat16)


def canary_time():
    t0 = time.perf_counter()
    mx.eval(CAN_X @ CAN_W.T)
    return (time.perf_counter() - t0) * 1e3


def main():
    t = build_transformer()
    inp = capture_block_inputs(t)
    blk = t.layers[BLOCK]
    orig_attn, orig_ff = blk.attention, blk.feed_forward
    args = [inp["x"], inp["attn_mask"], inp["freqs_cis"], inp["t_emb"]]
    RESULTS.mkdir(parents=True, exist_ok=True)

    def set_state(split):
        blk.attention = split[0] if split else orig_attn
        blk.feed_forward = split[1] if split else orig_ff

    def block_call():
        return blk(x=inp["x"], attn_mask=inp["attn_mask"], freqs_cis=inp["freqs_cis"], t_emb=inp["t_emb"])

    # One distinct compiled callable per arm: mx.compile caches by function object.
    fns, splits, ref_out = {}, {}, None
    for name, prec, share in ARMS:
        if prec is None:
            set_state(None)
        else:
            a = QL.AttentionSplit(orig_attn, CACHE, BLOCK, prec=prec, gpu_share=share)
            f = QL.FeedForwardSplit(orig_ff, CACHE, BLOCK, prec=prec, gpu_share=share)
            for m in (a.qkv, a.out, f.w13, f.w2):
                m.DRAIN = False  # mx.eval inside a compiled trace raises
            splits[name] = (a, f)
            set_state((a, f))
        fn = mx.compile(lambda u, m_, f_, te: blk(x=u, attn_mask=m_, freqs_cis=f_, t_emb=te))
        y = fn(*args)
        mx.eval(y)
        fns[name] = fn
        if ref_out is None:
            ref_out = np.array(y.astype(mx.float32))

    def cos(name):
        set_state(splits.get(name))
        y = np.array(block_call()[0].astype(mx.float32))
        return float(np.dot(y.ravel(), ref_out.ravel()) / (np.linalg.norm(y.ravel()) * np.linalg.norm(ref_out.ravel())))

    def cos_against(ref_name, name):
        outs = {}
        for n in (ref_name, name):
            set_state(splits.get(n))
            outs[n] = np.array(block_call()[0].astype(mx.float32)).ravel()
        a, b = outs[name], outs[ref_name]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    emit(
        dict(
            phase="correctness",
            eager_cosine_vs_base={n: round(cos(n), 7) for n, _, _ in ARMS},
            eager_cosine_vs_w8a16_50={n: round(cos_against("w8a16@50", n), 7) for n, _, _ in ARMS if n != "base"},
        )
    )

    best = {n: [] for n, _, _ in ARMS}
    windows = {n: [] for n, _, _ in ARMS}
    canary = []
    mm = subprocess.Popen(["macmon", "pipe", "-i", "1000"], stdout=open(MACMON, "w"), stderr=subprocess.DEVNULL)
    try:
        for rnd in range(ROUNDS):
            canary.append(canary_time())
            for name, _, _ in ARMS:
                fn = fns[name]
                ts = []
                w0 = time.time()
                for _ in range(ITERS):
                    t0 = time.perf_counter()
                    mx.eval(fn(*args))
                    ts.append((time.perf_counter() - t0) * 1e3)
                best[name].append(statistics.median(ts))
                windows[name].append((w0, time.time()))
            emit(
                dict(
                    phase="round",
                    round=rnd,
                    canary_ms=round(canary[-1], 1),
                    **{n: round(v[-1], 1) for n, v in best.items()},
                )
            )
    finally:
        mm.terminate()
        set_state(None)

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
            vals += [r["ane_power"] for r in mrecs if w0 <= r["_ts"] <= w1 and r.get("ane_power")]
        return dict(mean=round(sum(vals) / len(vals), 2), mx=round(max(vals), 2)) if vals else None

    base = [best["base"][i] for i in range(ROUNDS)]
    summary = {}
    for name, _, _ in ARMS:
        if name == "base":
            continue
        paired = [best[name][i] / base[i] for i in range(ROUNDS)]
        summary[name] = dict(
            med_ms=round(statistics.median(best[name]), 1),
            spread=round(max(best[name]) - min(best[name]), 1),
            ratio_vs_base=round(statistics.median(paired), 4),
            ane=ane_activity(name),
        )
    emit(
        dict(
            phase="summary",
            base_med_ms=round(statistics.median(base), 1),
            canary_med_ms=round(statistics.median(canary), 1),
            canary_max_ms=round(max(canary), 1),
            arms=summary,
        )
    )


def emit(record):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "block_q8_sweep.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"owner": f"3b2-sweep-{os.getpid()}", "pid": os.getpid(), "at": time.time()}).encode())
    try:
        main()
    finally:
        os.close(fd)
        try:
            os.unlink(LOCK)
        except FileNotFoundError:
            pass
