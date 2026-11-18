"""Fused vs separate w1/w3, head-to-head (interleaved arms).

Arms at T=4224, W8A16 per-channel int8:
  a_fused    20480x3840, one call               (--calls 1)
  b_separate 10240x3840, two back-to-back calls (--calls 2)
  c_overhead tiny model 64x3840x3840, one call  (bounds the per-call cost)
8 rounds, arms interleaved within each round. Emits results/fused_sep.jsonl.
"""

import os

import s3a
from s3a import MODELS, BenchLock, anebench, emit, gen_fixed

T = s3a.T_FIXED
ROUNDS = 8
ARMS = [
    ("a_fused", dict(shape=(T, 20480, 3840), calls=1, iters=30)),
    ("b_separate", dict(shape=(T, 10240, 3840), calls=2, iters=30)),
    ("c_overhead", dict(shape=(64, 3840, 3840), calls=1, iters=200)),
]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    models = {
        "a_fused": gen_fixed(MODELS, T, 20480, 3840, "c1"),
        "b_separate": gen_fixed(MODELS, T, 10240, 3840, "c1"),
        "c_overhead": gen_fixed(MODELS, 64, 3840, 3840, "c1"),
    }
    with BenchLock(owner=f"stage3a-fusedsep-{os.getpid()}"):
        for rnd in range(1, ROUNDS + 1):
            row = {"round": rnd}
            for arm, cfg in ARMS:
                r = anebench(models[arm], repeat=1, **cfg)
                row[arm] = {k: r[k] for k in ("median_ms", "p90_ms", "tflops", "calls")}
            emit("fused_sep", row)


if __name__ == "__main__":
    main()
