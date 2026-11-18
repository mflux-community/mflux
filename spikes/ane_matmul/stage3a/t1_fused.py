"""Test 1: fused ANE calls against separate calls, ANE alone.

QKV fused (11520x3840) against 3 separate 3840x3840 calls.
w1w3 fused (20480x3840) against 2 separate 10240x3840 calls.
Both precisions: W8A16 (c1) and W8A8 per-token (c1d). T = 4224.
"""

import s3a
from s3a import MODELS, BenchLock, anebench, emit, gen_fixed

T = s3a.T_FIXED
CASES = [
    # (label, fused(O, I, variant), separate(O, I, variant), n_calls)
    ("qkv_w8a16", (11520, 3840, "c1"), (3840, 3840, "c1"), 3),
    ("qkv_w8a8", (11520, 3840, "c1d"), (3840, 3840, "c1d"), 3),
    ("w1w3_w8a16", (20480, 3840, "c1"), (10240, 3840, "c1"), 2),
    ("w1w3_w8a8", (20480, 3840, "c1d"), (10240, 3840, "c1d"), 2),
]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    for label, fused, sep, n in CASES:
        for O, I, v in (fused, sep):
            gen_fixed(MODELS, T, O, I, v)
    with BenchLock(owner=f"stage3a-t1-{__import__('os').getpid()}"):
        for label, (fused_o, fused_i, fused_v), (sep_o, sep_i, sep_v), n in CASES:
            fused_path = MODELS / f"{fused_v}_{T}x{fused_o}x{fused_i}.mlpackage"
            sep_path = MODELS / f"{sep_v}_{T}x{sep_o}x{sep_i}.mlpackage"
            r_fused = anebench(fused_path, shape=(T, fused_o, fused_i), iters=50, repeat=3)
            r_sep = anebench(sep_path, shape=(T, sep_o, sep_i), iters=50, repeat=3)
            emit(
                "t1_fused",
                dict(
                    case=label,
                    fused_ms=r_fused["median_ms"],
                    sep_one_ms=r_sep["median_ms"],
                    sep_total_ms=r_sep["median_ms"] * n,
                    n_calls=n,
                    speedup=r_sep["median_ms"] * n / r_fused["median_ms"],
                    fused_tflops=r_fused["tflops"],
                    sep_tflops=r_sep["tflops"],
                ),
            )


if __name__ == "__main__":
    main()
