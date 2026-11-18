"""w1w3 recompile check (mini PRD, 2026-11-18).

Question: why did the standalone w1w3 package run 95-103 ms in t4b while the
same weights inside the multifunction package ran 53-55 ms, and the same
standalone measured 56 ms in t4?

Design: four arms, interleaved, 6 rounds, best-of-3 per round per arm.
  A  mf        mf_block.mlmodelc, function w1w3 (reference)
  B  sa_oldc   sa_w1w3.mlmodelc compiled earlier (the t4b artifact)
  C  sa_fresh  package rebuilt from the same deterministic weights, fresh compile
  D  sa_recom  the OLD package recompiled fresh (separates package from compile)

Weights: gen3a._synth_weights(20480, 3840, seed=102), identical to t4's build.
Records artifact sizes, weight-blob hashes, compute units, and ANE state probes
(3840x3840 c1 baseline) before and after the A/B.
"""

import hashlib
import os
import subprocess

import gen3a
import s3a
from s3a import MODELS, BenchLock, anebench, emit, package_bytes

T = s3a.T_FIXED
O, I = 20480, 3840
COMPILED = MODELS / "compiled"
FRESH = MODELS / "compiled_fresh"
BASELINE_MODEL = MODELS / "c1_4224x3840x3840.mlpackage"
ROUNDS = 6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def weight_blob(pkg):
    p = pkg / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    return p if p.exists() else None


def compile_fresh(pkg, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / (os.path.basename(str(pkg)).replace(".mlpackage", ".mlmodelc"))
    if not out.exists():
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(pkg), str(outdir)],
            check=True,
            capture_output=True,
            timeout=3600,
        )
    if not out.exists():
        raise RuntimeError(f"coremlcompiler produced no {out}")
    return out


def artifact_info(tag, mlmodelc):
    blob = mlmodelc / "coremldata.bin"
    emit(
        "recompile_w1w3",
        dict(
            phase="artifact",
            arm=tag,
            path=str(mlmodelc),
            bytes=package_bytes(mlmodelc),
            coremldata_sha16=sha256(blob) if blob.exists() else None,
        ),
    )


def probe_state(when):
    r = anebench(BASELINE_MODEL, shape=(4224, 3840, 3840), iters=30, repeat=1)
    state = "fast" if r["median_ms"] < 14 else "slow"
    emit("recompile_w1w3", dict(phase="state", when=when, baseline_3840sq_ms=r["median_ms"], state=state))
    return r["median_ms"]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    # Build phase (CPU only, no lock): fresh package from the same weights.
    fresh_pkg = MODELS / "sa_w1w3_fresh.mlpackage"
    if not fresh_pkg.exists():
        w, s = gen3a._synth_weights(O, I, seed=102)
        prog = gen3a.build_fixed(T, O, I, ksplit=1, a8=False, w_s=(w, s))
        gen3a.convert(prog, fresh_pkg)
    old_pkg = MODELS / "sa_w1w3.mlpackage"

    # Weight-blob identity between old and fresh packages.
    wb_old, wb_new = weight_blob(old_pkg), weight_blob(fresh_pkg)
    emit(
        "recompile_w1w3",
        dict(
            phase="weights",
            old_sha16=sha256(wb_old) if wb_old else None,
            fresh_sha16=sha256(wb_new) if wb_new else None,
            identical=bool(wb_old and wb_new and sha256(wb_old) == sha256(wb_new)),
        ),
    )

    # Compile phase (CPU only): fresh compiles of both packages.
    arms = {
        "A_mf": (COMPILED / "mf_block.mlmodelc", "w1w3"),
        "B_sa_oldc": (COMPILED / "sa_w1w3.mlmodelc", None),
        "C_sa_fresh": (compile_fresh(fresh_pkg, FRESH), None),
        "D_sa_recom": (compile_fresh(old_pkg, FRESH / "recom"), None),
    }
    for tag, (path, _fn) in arms.items():
        artifact_info(tag, path)

    # Timed phase: state probe, interleaved A/B, state probe. Under the lock.
    with BenchLock(owner=f"stage3a-recompile-w1w3-{os.getpid()}"):
        probe_state("before")
        for rnd in range(1, ROUNDS + 1):
            row = {"round": rnd}
            for tag, (path, fn) in arms.items():
                r = anebench(path, shape=(T, O, I), function=fn, iters=30, repeat=3)
                row[tag + "_ms"] = r["median_ms"]
                row[tag + "_units"] = r["units"]
            emit("recompile_w1w3", dict(phase="ab", **row))
        probe_state("after")


if __name__ == "__main__":
    main()
