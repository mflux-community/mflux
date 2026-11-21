"""Pre-build every model the Stage 3a tests need. Model generation is cached and not timed.

Safe to run while the machine is busy; the timed test scripts skip existing models.
"""

import gen3a
import s3a
from s3a import MODELS, gen_fixed

T = s3a.T_FIXED


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    # t1: fused and separate, W8A16 and W8A8.
    for O, I, v in [
        (11520, 3840, "c1"),
        (3840, 3840, "c1"),
        (20480, 3840, "c1"),
        (10240, 3840, "c1"),
        (11520, 3840, "c1d"),
        (3840, 3840, "c1d"),
        (20480, 3840, "c1d"),
        (10240, 3840, "c1d"),
    ]:
        gen_fixed(MODELS, T, O, I, v)
        print("built", f"{v}_{T}x{O}x{I}", flush=True)

    # t2: ANE-side column counts per shape and split.
    shapes = [
        ("o", 3840, 3840, "c1"),
        ("qkv", 11520, 3840, "c1"),
        ("w1w3", 20480, 3840, "c1"),
        ("w2", 3840, 10240, "c4"),
    ]
    for _, O, I, v in shapes:
        for share in (0.4, 0.5, 0.55, 0.6, 1.0):
            o_ane = max(64, int(round(O * share / 64)) * 64)
            gen_fixed(MODELS, T, o_ane, I, v)
            print("built", f"{v}_{T}x{o_ane}x{I}", flush=True)

    # t3: enumerated models (fixed per-token models are built by t3 itself).
    for name, O, I, ksplit in [("w1", 10240, 3840, 1), ("w2", 3840, 10240, 4)]:
        p = MODELS / f"enum_{name}.mlpackage"
        if not p.exists():
            prog, ct_inputs = gen3a.build_enum(O, I, [4608, 6528, 8320, 16128], ksplit=ksplit)
            gen3a.convert(prog, p, inputs=ct_inputs)
        print("built", p.name, flush=True)
        for t in (4608, 6528, 8320, 16128):
            gen_fixed(MODELS, t, O, I, "c1" if ksplit == 1 else "c4")
            print("built", f"fixed_{name}_{t}", flush=True)

    # t4: multifunction package and standalones are built inside t4_multifn.py; trigger it here.
    import t4_multifn

    mf_path = MODELS / "mf_block.mlpackage"
    if not mf_path.exists():
        print("building multifunction package (large)...", flush=True)
    t4_multifn.build_packages()
    print("built mf_block + standalones", flush=True)


if __name__ == "__main__":
    main()
