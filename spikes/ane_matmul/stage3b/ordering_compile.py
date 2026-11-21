"""3b-0: does the ordering workaround survive mx.compile?

Real Z-Image transformer architecture with random bf16 weights (no checkpoint),
real predict path, block 15's four fused call sites split to the ANE.
Six timed configurations, interleaved 3 rounds each, best median per config:

  eager_base / compiled_base          no split
  eager_split_{hack,nohack}           split, mx.compile off
  compiled_split_{hack,nohack}        split, mx.compile on (the production path)

Gate (PRD 3b-0): the compiled split step must save at least 90% of what the
eager split step saves, with the hack on both sides. Random weights make the
absolute activations unrealistically large; timing and overlap are unaffected.

Run under BENCH.lock + watchdog.sh, foreground session, after the throttle
probe passes. macmon logs the ANE rail across the whole lock session.
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
import split_layers as SL  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer  # noqa: E402
from mflux.models.z_image.variants.z_image import ZImage  # noqa: E402

BLOCK = 15
ROUNDS = 3
ITERS = 5
CACHE = Path("/tmp/ane3a/cache3b")
RESULTS = HERE / "results"
MACMON = RESULTS / "ordering_compile_macmon.jsonl"


def build_transformer():
    """Random-weight transformer with bf16 parameters, like a loaded Z-Image model."""
    t = ZImageTransformer()
    t.update(
        tree_map(
            lambda v: v.astype(mx.bfloat16) if isinstance(v, mx.array) and v.dtype != mx.complex64 else v,
            t.parameters(),
        )
    )
    mx.eval(t.parameters())
    return t


def make_inputs():
    rng = np.random.default_rng(11)
    x = mx.array(rng.standard_normal((16, 1, 128, 128)).astype(np.float32)).astype(mx.bfloat16)
    cap = mx.array(rng.standard_normal((32, 2560)).astype(np.float32)).astype(mx.bfloat16)
    timestep = mx.array(0.5, mx.float32)
    sigmas = mx.zeros((8,), mx.float32)
    return x, cap, timestep, sigmas


def time_step(run):
    run()
    mx.eval(run())
    meds = []
    for _ in range(ROUNDS):
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            mx.eval(run())
            ts.append(time.perf_counter() - t0)
        meds.append(statistics.median(ts))
    return min(meds), meds


def emit(record):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "ordering_compile.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def main():
    transformer = build_transformer()
    x, cap, timestep, sigmas = make_inputs()

    def eager_run():
        return transformer(timestep=timestep, x=x, cap_feats=cap, sigmas=sigmas)

    compiled = ZImage._predict(transformer)

    def compiled_run():
        return compiled(x, timestep, sigmas, cap, None, 0.0)

    eager_base, eb_all = time_step(eager_run)
    compiled_base, cb_all = time_step(compiled_run)
    emit(
        dict(
            phase="base", eager_base_s=eager_base, eager_all=eb_all, compiled_base_s=compiled_base, compiled_all=cb_all
        )
    )

    # Patch block 15: four fused call sites.
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    blk = transformer.layers[BLOCK]
    attn_split = SL.AttentionSplit(blk.attention, CACHE, BLOCK)
    ff_split = SL.FeedForwardSplit(blk.feed_forward, CACHE, BLOCK)
    blk.attention = attn_split
    blk.feed_forward = ff_split
    build_s = time.perf_counter() - t0

    def set_hack(v):
        attn_split.set_order_hack(v)
        ff_split.set_order_hack(v)

    out = dict(phase="split_build", build_s=round(build_s, 1))
    emit(out)

    results = {}
    for name, base in [("eager", eager_base), ("compiled", compiled_base)]:
        for hack in (True, False):
            set_hack(hack)
            if name == "eager":
                run = eager_run
            else:
                # mx.compile caches per callable: build a fresh compiled function
                # for each hack state, or the trace from the first state is reused.
                fresh = ZImage._predict(transformer)

                def run(_f=fresh):
                    return _f(x, timestep, sigmas, cap, None, 0.0)

            med, all_meds = time_step(run)
            results[f"{name}_{'hack' if hack else 'nohack'}"] = med
            emit(
                dict(
                    phase="split",
                    config=f"{name}_{'hack' if hack else 'nohack'}",
                    step_s=med,
                    all=all_meds,
                    base_s=base,
                    saved_s=base - med,
                )
            )

    eager_saved = eager_base - results["eager_hack"]
    compiled_saved = compiled_base - results["compiled_hack"]
    ratio = compiled_saved / eager_saved if eager_saved > 0 else None
    gate = "PASS" if ratio is not None and ratio >= 0.9 else "FAIL"
    emit(
        dict(
            phase="gate",
            eager_saved_s=round(eager_saved, 3),
            compiled_saved_s=round(compiled_saved, 3),
            ratio=ratio,
            verdict=gate,
            nohack_eager_saved=round(eager_base - results["eager_nohack"], 3),
            nohack_compiled_saved=round(compiled_base - results["compiled_nohack"], 3),
        )
    )
    recs = [json.loads(l) for l in open(MACMON) if l.strip()]
    ap = sorted(r.get("ane_power", 0) for r in recs)
    emit(
        dict(
            phase="canary",
            n=len(recs),
            ane_power_med=ap[len(ap) // 2] if ap else None,
            ane_power_max=ap[-1] if ap else None,
        )
    )


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = "/Users/ianscrivener/MFlux/mflux-upstream/spikes/BENCH.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"owner": f"3b0-ordering-{os.getpid()}", "pid": os.getpid(), "at": time.time()}).encode())
    mm = subprocess.Popen(["macmon", "pipe", "-i", "2000"], stdout=open(MACMON, "w"), stderr=subprocess.DEVNULL)
    try:
        main()
    finally:
        mm.terminate()
        os.close(fd)
        try:
            os.unlink(lock)
        except FileNotFoundError:
            pass
