"""Stage 3c weightless programs: build once per (Mpad,N,K), persist across runs.

Cache lives under ~/Library/Caches/mflux/ane_wl (NOT /tmp: macOS cleans it and
a missing program looks like a measurement failure). Key embeds OS major and
coremltools version because compiled artifacts are OS-format sensitive.
Design follows ~/zCode_26_11/drawthings-ane-research/DRAWTHINGS-ANE-FINDINGS.md
(liuliu/ccv, BSD-3-Clause); written from the description, no source copied.
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("MFLUX_WL_CACHE", str(Path.home() / "Library" / "Caches" / "mflux" / "ane_wl")))
KLIMIT = 2688


def k_slices(K):
    if K <= KLIMIT:
        return 1
    return 2 if K / 2 <= KLIMIT else 4


def _key(Mpad, N, K):
    import platform

    import coremltools as ct

    os_major = platform.mac_ver()[0].split(".")[0]
    return f"wl_{Mpad}_{N}_{K}_os{os_major}_ct{ct.__version__}"


def build_program(Mpad, N, K):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    target = ct.target.iOS26 if hasattr(ct.target, "iOS26") else ct.target.iOS18
    ns = k_slices(K)
    ks = K // ns
    c = np.float16(1.0 / np.sqrt(K))

    @mb.program(
        input_specs=[mb.TensorSpec((1, 1, N, K), dtype=types.int8), mb.TensorSpec((1, 1, K, Mpad), dtype=types.int8)],
        opset_version=target,
    )
    def prog(w, x):
        wd = mb.dequantize(input=w, scale=c)
        xd = mb.dequantize(input=x, scale=c)
        y = None
        for i in range(ns):
            if ns == 1:
                wm, xm = wd, xd
            else:
                wm = mb.slice_by_size(x=wd, begin=[0, 0, 0, i * ks], size=[1, 1, N, ks])
                xm = mb.slice_by_size(x=xd, begin=[0, 0, i * ks, 0], size=[1, 1, ks, Mpad])
            part = mb.matmul(x=wm, y=xm)
            y = part if y is None else mb.add(x=y, y=part)
        return mb.identity(x=y, name="y")

    ci = [ct.TensorType(name=n) for n in ("w", "x")]
    return ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=ci,
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


_models = {}


def get_model(ext, Mpad, N, K):
    """Return a loaded WLModel, converting and compiling at most once per key."""
    key = _key(Mpad, N, K)
    if key in _models:
        return _models[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    pkg = CACHE / f"{key}.mlpackage"
    compiled = CACHE / f"{key}.mlmodelc"
    if not pkg.exists():
        t0 = time.perf_counter()
        m = build_program(Mpad, N, K)
        m.save(str(pkg))
        _ = round(time.perf_counter() - t0, 1)  # convert cost, for the cold-run log
    model = ext.WLModel(str(pkg), str(compiled), Mpad, N, K)
    _models[key] = model
    return model


def prune_caches(keep_prefix=None):
    """Remove cached programs; used by tests to force cold paths."""
    removed = 0
    for p in CACHE.glob("wl_*"):
        if keep_prefix and keep_prefix not in p.name:
            continue
        shutil.rmtree(p, ignore_errors=True)
        removed += 1
    return removed
