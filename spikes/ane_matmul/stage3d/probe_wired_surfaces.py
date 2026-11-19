"""B, step 1: is T4's wired growth charged per distinct weight surface, or per program?

Stage 3d T4 grew wired memory +4.86 GB, against a +1 GB allowance, while 3c's column split drove
wired DOWN during patching. The difference is that a row split must keep both a bf16 copy (GPU rows)
and an int8 copy (ANE rows) of every weight. This measures the kernel side of that directly:

  arm "distinct"  -- one program, k different host weight buffers, each first-used once.
  arm "reused"    -- one program, ONE weight buffer, new weights blitted into it per call.

If wired grows with k in "distinct" and stays flat in "reused", the +4.86 GB is per-surface wiring
and a single shared surface fixes it -- the question is then what the per-call copy costs, which this
also reports. If wired grows the same way in both, the cost is the program, not the surfaces.

Synthetic weights only. No timed step, no real weights.
"""

import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx  # noqa: E402,F401  -- must load before _ane_linear, which needs its NB_DOMAIN
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ane_linear"))  # BridgeBackend does this lazily; here we import ext directly

from mlp_program import DIM, HIDDEN, MlpSpec, program_scales, quantize_per_row  # noqa: E402


def _import_ext():
    # Inside a function so import sorting cannot move this above the mlx import above it.
    import _ane_linear as ext

    return ext


ext = _import_ext()

M = 2048
SIZES = [1, 2, 4, 8, 16, 34]
CACHE = HERE / "cache" / "probeWired"
COMPILED = Path.home() / "Library/Caches/mflux/ane_wl3d_probeWired"
CACHE.mkdir(parents=True, exist_ok=True)
COMPILED.mkdir(parents=True, exist_ok=True)


def wired_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = re.search(r"page size of (\d+)", out)
    wired = re.search(r"Pages wired down:\s+(\d+)", out)
    return int(wired.group(1)) * int(page.group(1)) / 2**30


def swap_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else float("nan")


def make_set(seed: int):
    r = np.random.default_rng(seed)
    w1 = r.normal(0, 0.05, (HIDDEN, DIM)).astype(np.float32)
    w3 = r.normal(0, 0.05, (HIDDEN, DIM)).astype(np.float32)
    w2 = r.normal(0, 0.05, (DIM, HIDDEN)).astype(np.float32)
    (q1, s1), (q3, s3), (q2, s2) = quantize_per_row(w1), quantize_per_row(w3), quantize_per_row(w2)
    spec = MlpSpec(mpad=M, scale_mode="output")
    sc = program_scales(spec, s1, s3, s2)
    return spec, q1, q3, q2, sc


def load(spec):
    from mlp_program import convert_to_mlpackage

    pkg = CACHE / f"{spec.name}.mlpackage"
    if not pkg.exists():
        convert_to_mlpackage(spec, str(CACHE))
    return ext.wl_fused_load(
        str(pkg),
        str(COMPILED / spec.name),
        ["x_rows", "w1", "w3", "w2", "s1", "s3", "s2"],
        [[1, M, DIM], [HIDDEN, DIM], [HIDDEN, DIM], [DIM, HIDDEN], [HIDDEN], [HIDDEN], [DIM]],
        [0, 1, 1, 1, 0, 0, 0],
        "y_rows",
        [1, M, DIM],
    )


x = np.ascontiguousarray(np.random.default_rng(1).normal(0, 1.0, (1, M, DIM)).astype(np.float16))
out = np.zeros((1, M, DIM), dtype=np.float16)

print("arm distinct: one program, k separate host weight buffers")
print(f"{'k':>4} {'wired GB':>10} {'Δ vs k=1':>10} {'host int8 GB':>13} {'ms/call':>9} {'swap MB':>9}")
base = None
host_gb = 0.0
models = []
for k in SIZES:
    spec, q1, q3, q2, sc = make_set(100 + k)
    if not models:
        model = load(spec)
    ins = [x, q1, q3, q2, sc["s1"], sc["s3"], sc["s2"]]
    ptrs = [a.ctypes.data for a in ins]
    ext.wl_fused_predict(model, ptrs, out.ctypes.data)
    ext.wl_fused_predict(model, ptrs, out.ctypes.data)  # second call wires this surface set
    t0 = time.perf_counter()
    ext.wl_fused_predict(model, ptrs, out.ctypes.data)
    ms = (time.perf_counter() - t0) * 1e3
    host_gb += 3 * HIDDEN * DIM  # int8 bytes per set
    w = wired_gb()
    base = w if base is None else base
    models.append((q1, q3, q2, sc))  # keep buffers alive: wired memory tracks live surfaces
    print(f"{k:>4} {w:>10.2f} {w - base:>10.2f} {host_gb / 2**30:>13.2f} {ms:>9.2f} {swap_mb():>9.2f}")

print("\narm reused: one program, ONE weight buffer set, new weights blitted in per call")
print(f"{'calls':>6} {'wired GB':>10} {'Δ vs first':>11} {'copy ms':>9} {'predict ms':>11} {'swap MB':>9}")
spec, q1, q3, q2, sc = make_set(999)
surfaces = [
    np.empty_like(q1),
    np.empty_like(q3),
    np.empty_like(q2),
    np.empty_like(sc["s1"]),
    np.empty_like(sc["s3"]),
    np.empty_like(sc["s2"]),
]
model2 = load(spec)
w0 = None
copies, predicts = [], []
for i in range(len(SIZES) * 3):
    _, n_q1, n_q3, n_q2, n_sc = make_set(200 + i)  # one fresh weight set per call
    t0 = time.perf_counter()
    for dst, src in zip(surfaces, [n_q1, n_q3, n_q2, n_sc["s1"], n_sc["s3"], n_sc["s2"]], strict=True):
        np.copyto(dst, src)
    copies.append((time.perf_counter() - t0) * 1e3)
    ins = [x, *surfaces]
    t1 = time.perf_counter()
    ext.wl_fused_predict(model2, [a.ctypes.data for a in ins], out.ctypes.data)
    predicts.append((time.perf_counter() - t1) * 1e3)
    w = wired_gb()
    w0 = w if w0 is None else w0
    if i % 6 == 5:
        print(
            f"{i + 1:>6} {w:>10.2f} {w - w0:>11.2f} {statistics.median(copies):>9.2f} "
            f"{statistics.median(predicts):>11.2f} {swap_mb():>9.2f}"
        )
    del n_q1, n_q3, n_q2, n_sc, ins

print("\nIf 'distinct' wired rises with k while 'reused' stays flat, the +4.86 GB is per-surface")
print("wiring and one shared surface fixes it -- at the copy cost printed above.")
