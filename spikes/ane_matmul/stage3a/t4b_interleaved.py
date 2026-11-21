"""Test 4b: interleaved multifunction-vs-standalone A/B, to beat the bimodal ANE state.

The machine alternates between a fast and a slow ANE state on the scale of
minutes, so back-to-back best-of-3 blocks compare different states. This script
alternates single mf/sa measurements within seconds and reports both the best
of each and the per-pair ratios.
"""

import os

import s3a
from s3a import MODELS, BenchLock, anebench, emit

T = s3a.T_FIXED
COMPILED = MODELS / "compiled"
PAIRS = [  # (layer, a8, O, I)
    ("qkv", False, 11520, 3840),
    ("w1w3", False, 20480, 3840),
    ("w2", False, 3840, 10240),
    ("o", False, 3840, 3840),
]
ROUNDS = 6


def main():
    with BenchLock(owner=f"stage3a-t4b-{os.getpid()}"):
        for layer, a8, O, I in PAIRS:
            fname = layer + ("_a8" if a8 else "")
            mf_c = COMPILED / "mf_block.mlmodelc"
            sa_c = COMPILED / f"sa_{fname}.mlmodelc"
            mf_runs, sa_runs = [], []
            for _ in range(ROUNDS):
                mf_runs.append(anebench(mf_c, shape=(T, O, I), function=fname, iters=30)["median_ms"])
                sa_runs.append(anebench(sa_c, shape=(T, O, I), iters=30)["median_ms"])
            pair_ratios = [s / m for m, s in zip(mf_runs, sa_runs)]
            emit(
                "t4b_interleaved",
                dict(
                    layer=layer,
                    a8=a8,
                    mf_best_ms=min(mf_runs),
                    sa_best_ms=min(sa_runs),
                    best_ratio=min(sa_runs) / min(mf_runs),
                    median_pair_ratio=sorted(pair_ratios)[len(pair_ratios) // 2],
                    mf_runs=[round(v, 2) for v in mf_runs],
                    sa_runs=[round(v, 2) for v in sa_runs],
                ),
            )


if __name__ == "__main__":
    main()
