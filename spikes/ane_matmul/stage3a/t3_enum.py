"""Test 3: one function with enumerated token counts against fixed-shape models.

Token tiers {4608, 6528, 8320, 16128}. Layer shapes: w1 (10240x3840) and w2
(3840x10240, K-split 4). Reports ANE placement, per-tier speed against the
fixed-shape model, and load behaviour. Synthetic weights.
"""

import os

import gen3a
import s3a
from s3a import MODELS, BenchLock, anebench, aneplan, emit, gen_fixed

TOKENS = [4608, 6528, 8320, 16128]
LAYERS = [("w1", 10240, 3840, 1, "c1"), ("w2", 3840, 10240, 4, "c4")]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    # Build phase (not timed): enumerated and fixed models.
    enum_paths = {}
    for name, O, I, ksplit, fixed_variant in LAYERS:
        p = MODELS / f"enum_{name}.mlpackage"
        if not p.exists():
            prog, ct_inputs = gen3a.build_enum(O, I, TOKENS, ksplit=ksplit)
            gen3a.convert(prog, p, inputs=ct_inputs)
        enum_paths[name] = p
        for T in TOKENS:
            gen_fixed(MODELS, T, O, I, fixed_variant)

    # Placement (not timed). Skip if an earlier run already recorded the plans.
    prior = (s3a.RESULTS / "t3_enum.jsonl").read_text() if (s3a.RESULTS / "t3_enum.jsonl").exists() else ""
    if '"phase": "plan"' not in prior:
        for name, O, I, ksplit, fixed_variant in LAYERS:
            emit("t3_enum", dict(phase="plan", layer=name, kind="enum", plan=aneplan(enum_paths[name])))
            emit(
                "t3_enum",
                dict(
                    phase="plan",
                    layer=name,
                    kind="fixed",
                    plan=aneplan(MODELS / f"{fixed_variant}_{TOKENS[0]}x{O}x{I}.mlpackage"),
                ),
            )

    with BenchLock(owner=f"stage3a-t3-{os.getpid()}"):
        for name, O, I, ksplit, fixed_variant in LAYERS:
            for T in TOKENS:
                fixed = anebench(MODELS / f"{fixed_variant}_{T}x{O}x{I}.mlpackage", shape=(T, O, I), iters=30, repeat=3)
                try:
                    enum = anebench(enum_paths[name], shape=(T, O, I), iters=30, repeat=3)
                    enum_res = dict(
                        enum_ms=enum["median_ms"],
                        enum_tflops=enum["tflops"],
                        enum_load_s=enum["load_s"],
                        enum_error=None,
                    )
                except RuntimeError as e:
                    enum_res = dict(enum_ms=None, enum_tflops=None, enum_load_s=None, enum_error=str(e)[:400])
                emit(
                    "t3_enum",
                    dict(
                        phase="bench",
                        layer=name,
                        T=T,
                        fixed_ms=fixed["median_ms"],
                        fixed_tflops=fixed["tflops"],
                        ratio=(fixed["median_ms"] / enum_res["enum_ms"]) if enum_res["enum_ms"] else None,
                        **enum_res,
                    ),
                )


if __name__ == "__main__":
    main()
