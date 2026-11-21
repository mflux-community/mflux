"""Test 1b: fused against separate in the production W8A8 form (c1p: int8 input, token scale after the matmul).

t1 covered W8A16 (c1) and the in-graph dynamic W8A8 (c1d). Test 7 showed the
production W8A8 form is c1p, so the fused-vs-separate decision needs c1p numbers.
"""

import os

import s3a
from s3a import MODELS, BenchLock, anebench, emit, gen_fixed

T = s3a.T_FIXED
CASES = [
    ("qkv_w8a8_pre", (11520, 3840), (3840, 3840), 3),
    ("w1w3_w8a8_pre", (20480, 3840), (10240, 3840), 2),
]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    for _, fused, sep, n in CASES:
        for O, I in (fused, sep):
            gen_fixed(MODELS, T, O, I, "c1p")
    with BenchLock(owner=f"stage3a-t1b-{os.getpid()}"):
        for label, (fused_o, fused_i), (sep_o, sep_i), n in CASES:
            fused_path = MODELS / f"c1p_{T}x{fused_o}x{fused_i}.mlpackage"
            sep_path = MODELS / f"c1p_{T}x{sep_o}x{sep_i}.mlpackage"
            r_fused = anebench(fused_path, shape=(T, fused_o, fused_i), iters=50, repeat=3, int8in=True)
            r_sep = anebench(sep_path, shape=(T, sep_o, sep_i), iters=50, repeat=3, int8in=True)
            emit(
                "t1b_fused_pre",
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
