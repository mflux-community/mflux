"""Test 4: multifunction package (one DiT block) against standalone packages.

8 functions at T=4224: qkv, o, w1w3, w2 in W8A16, and W8A8 twins that share the
same weight arrays. Measures package size (weight deduplication), precompiled
load time per function, and runtime speed per function against standalones.
"""

import os
import subprocess

import gen3a
import s3a
from s3a import MODELS, BenchLock, anebench, aneplan, emit, package_bytes

T = s3a.T_FIXED
LAYERS = {"qkv": (11520, 3840, 1), "o": (3840, 3840, 1), "w1w3": (20480, 3840, 1), "w2": (3840, 10240, 4)}
COMPILED = MODELS / "compiled"


def compile_c(mlpackage):
    """Precompile one .mlpackage into COMPILED/<name>.mlmodelc; return the path."""
    COMPILED.mkdir(parents=True, exist_ok=True)
    out = COMPILED / (os.path.basename(str(mlpackage)).replace(".mlpackage", ".mlmodelc"))
    if not out.exists():
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(COMPILED)],
            check=True,
            capture_output=True,
            timeout=3600,
        )
    if not out.exists():
        raise RuntimeError(f"coremlcompiler produced no {out}")
    return out


def build_packages():
    """Build the multifunction package and the 8 standalones from identical weight arrays. Cached."""
    weights = {}
    for i, (name, (O, I, ksplit)) in enumerate(LAYERS.items()):
        weights[name] = gen3a._synth_weights(O, I, seed=100 + i)

    mf_path = MODELS / "mf_block.mlpackage"
    if not mf_path.exists():
        from coremltools.converters.mil import mil

        prog = mil.Program()
        first = None
        for name, (O, I, ksplit) in LAYERS.items():
            w, s = weights[name]
            _, chunks = gen3a._k_chunks(w, I, ksplit)
            for a8 in (False, True):
                fname = name + ("_a8" if a8 else "")

                def make(a8=a8, chunks=chunks, s=s, I=I, ksplit=ksplit):
                    from coremltools.converters.mil import Builder as mb
                    from coremltools.converters.mil.mil import types

                    @mb.function(
                        input_specs=[mb.TensorSpec((1, I, 1, T), dtype=types.fp16)], opset_version=gen3a.TARGET
                    )
                    def f(x):
                        if a8:
                            x = gen3a._a8_dyn(x)
                        return mb.identity(x=gen3a._conv_body(x, chunks, s, ksplit), name="y")

                    return f

                prog.add_function(fname, make())
                if first is None:
                    first = fname
        prog.default_function_name = first
        prog.export_as_multifunction = True
        gen3a.convert(prog, mf_path)

    sa_paths = {}
    for a8 in (False, True):
        for name, (O, I, ksplit) in LAYERS.items():
            tag = f"sa_{name}{'_a8' if a8 else ''}"
            p = MODELS / f"{tag}.mlpackage"
            if not p.exists():
                w, s = weights[name]
                prog = gen3a.build_fixed(T, O, I, ksplit=ksplit, a8=a8, w_s=(w, s))
                gen3a.convert(prog, p)
            sa_paths[(name, a8)] = p
    return mf_path, sa_paths


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    mf_path, sa_paths = build_packages()

    sizes = {"mf": package_bytes(mf_path)}
    for (name, a8), p in sa_paths.items():
        sizes[f"sa_{name}{'_a8' if a8 else ''}"] = package_bytes(p)
    a16_total = sum(v for k, v in sizes.items() if k.startswith("sa_") and not k.endswith("_a8"))
    all_total = sum(v for k, v in sizes.items() if k.startswith("sa_"))
    emit(
        "t4_multifn",
        dict(
            phase="size",
            mf_bytes=sizes["mf"],
            standalone_a16_bytes=a16_total,
            standalone_all_bytes=all_total,
            sizes=sizes,
        ),
    )

    mf_c = compile_c(mf_path)
    sa_c = {k: compile_c(p) for k, p in sa_paths.items()}

    emit("t4_multifn", dict(phase="plan", function="qkv", plan=aneplan(mf_c, function="qkv")))
    emit("t4_multifn", dict(phase="plan", function="qkv_a8", plan=aneplan(mf_c, function="qkv_a8")))

    with BenchLock(owner=f"stage3a-t4-{os.getpid()}"):
        for name, (O, I, ksplit) in LAYERS.items():
            for a8 in (False, True):
                fname = name + ("_a8" if a8 else "")
                mf_r = anebench(mf_c, shape=(T, O, I), function=fname, iters=30)
                sa_r = anebench(sa_c[(name, a8)], shape=(T, O, I), iters=30)
                emit(
                    "t4_multifn",
                    dict(
                        phase="bench",
                        layer=name,
                        a8=a8,
                        mf_ms=mf_r["median_ms"],
                        sa_ms=sa_r["median_ms"],
                        mf_load_s=mf_r["load_s"],
                        sa_load_s=sa_r["load_s"],
                        mf_first_call_s=mf_r["first_call_s"],
                        sa_first_call_s=sa_r["first_call_s"],
                        ratio=sa_r["median_ms"] / mf_r["median_ms"],
                    ),
                )


if __name__ == "__main__":
    main()
