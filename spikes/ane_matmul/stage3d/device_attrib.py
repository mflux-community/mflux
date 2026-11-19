"""Did the fused MLP actually run on the Neural Engine?

The compute plan reports the PREFERRED device; Core ML can still fall back at runtime, and a
~4.7 TFLOPS marginal rate is also what an M4 CPU fp16 path looks like. This samples ANE power
during repeated calls and compares against an idle baseline and against the same op forced to
CPU via ComputeUnit.ALL/ CPU only, so a low-power result is attributable rather than guessed.
"""

import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ane_linear"))
from mlp_program import DIM, HIDDEN, MlpSpec  # noqa: E402
from rowsplit import prepare_block_weights  # noqa: E402


class Power:
    def __init__(self):
        self.samples = []
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(["macmon", "pipe", "-s", "0", "-i", "200"], stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass

    def __exit__(self, *exc):
        self.proc.terminate()

    def series(self, key):
        return [r[key] for r in self.samples if key in r and r[key] is not None]

    def keys(self):
        return sorted(self.samples[0].keys()) if self.samples else []


import coremltools as ct  # noqa: E402

rng = np.random.default_rng(0)
w1 = rng.normal(0, 0.02, (HIDDEN, DIM)).astype(np.float32)
w3 = rng.normal(0, 0.02, (HIDDEN, DIM)).astype(np.float32)
w2 = rng.normal(0, 0.02, (DIM, HIDDEN)).astype(np.float32)
spec = MlpSpec(mpad=2048)
weights = prepare_block_weights(spec, w1, w3, w2)
x = np.ascontiguousarray(rng.normal(0, 1, (1, 2048, DIM)).astype(np.float16))

pkg = HERE / "cache" / f"{spec.name}.mlpackage"
compiled = Path.home() / "Library/Caches/mflux/ane_wl3d" / spec.name
print("compiled model exists:", compiled.exists())

import _ane_linear as ext  # noqa: E402

model = ext.wl_fused_load(
    str(pkg),
    str(compiled),
    ["x_rows", "w1", "w3", "w2", "s1", "s3", "s2"],
    [[1, 2048, DIM], [HIDDEN, DIM], [HIDDEN, DIM], [DIM, HIDDEN], [HIDDEN], [HIDDEN], [DIM]],
    [0, 1, 1, 1, 0, 0, 0],
    ["y_rows"],
    [[1, 2048, DIM]],
    [0],
)
out = np.zeros((1, 2048, DIM), dtype=np.float16)
ptrs = [
    a.ctypes.data
    for a in (x, weights.w1, weights.w3, weights.w2, weights.scales["s1"], weights.scales["s3"], weights.scales["s2"])
]


def burst(label, fn, n=25):
    time.sleep(1.5)  # let the sampler cover the idle gap before the burst
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1e3
    print(f"  {label}: {dt:.2f} ms/call")
    return dt


print("\nmacmon keys:", Power().keys())

with Power() as p:
    base = len(p.samples)
    burst("warm", lambda: ext.wl_fused_predict(model, ptrs, [out.ctypes.data]), 3)
    gap = len(p.samples)
    burst("fused MLP M=2048", lambda: ext.wl_fused_predict(model, ptrs, [out.ctypes.data]), 25)
    end = len(p.samples)

print("\ncopybacks:", model.copybacks())
ane = p.series("ane_power")
gpu = p.series("gpu_power")
cpu = p.series("cpu_power")
print(f"samples: total={len(p.samples)}")
if ane:
    a = np.array(ane)
    print(
        f"ANE W: idle-window median={statistics.median(a[: max(1, base + 5)]):.2f} "
        f"burst median={statistics.median(a[gap:end]):.2f} max={a.max():.2f}"
    )
if gpu:
    print(f"GPU W: burst median={statistics.median(np.array(gpu)[gap:end]):.2f}")
if cpu:
    print(f"CPU W: burst median={statistics.median(np.array(cpu)[gap:end]):.2f}")

# Control: same program through coremltools on CPU only, to show what a non-ANE rate looks like.
print("\n--- control: coremltools predict, CPU_ONLY vs CPU_AND_NE ---")
feed = {
    "x_rows": x,
    "w1": weights.w1,
    "w3": weights.w3,
    "w2": weights.w2,
    "s1": weights.scales["s1"],
    "s3": weights.scales["s3"],
    "s2": weights.scales["s2"],
}
for cu, name in ((ct.ComputeUnit.CPU_ONLY, "CPU_ONLY"), (ct.ComputeUnit.CPU_AND_NE, "CPU_AND_NE")):
    ctl = ct.models.MLModel(str(pkg), compute_units=cu)
    ctl.predict(feed)
    with Power() as p2:
        ms = burst(name, lambda: ctl.predict(feed), 5)
    a2 = p2.series("ane_power")
    ane_w = statistics.median(a2) if a2 else float("nan")
    print(f"  {name}: {ms:.2f} ms/call, ANE W median={ane_w:.2f}")
