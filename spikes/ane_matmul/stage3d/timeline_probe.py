"""In-process timeline of arm C per layer: when the ANE call starts/ends vs when the GPU rows MLP finishes.

All times in ms from the start of the layer (t0 = call entry). Diagnostic only (no lock, not a timing claim).
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlx.core as mx  # noqa: E402
from eager_bench import load_real_blocks, make_epilogue, make_gpu_mlp, make_inputs, make_pre, real_ane_fns  # noqa: E402
from mlp_program import MlpSpec, plan_split  # noqa: E402

_, blocks = load_real_blocks(0, 6)
x, mask, freqs, t = make_inputs(4224, 3840)
m = plan_split(4224, 0.5)[0]
spec = MlpSpec(mpad=m, scale_mode="output", range_fix=True)
fns, gains, _ = real_ane_fns(blocks, spec, HERE / "cache")
pool = ThreadPoolExecutor(max_workers=1)
rows = []


def layer(i, b, fn, g, x):
    pre, gpu, epi = make_pre(b, m), make_gpu_mlp(b), make_epilogue(b, mx.array(g))
    log = {}

    def ane(xn):
        log["ane_start"] = time.perf_counter()
        r = fn(xn)
        log["ane_end"] = time.perf_counter()
        return r

    def run(x):
        t0 = time.perf_counter()
        x1, ha, hg, gate = pre(x, mask, freqs, t)
        mx.async_eval(x1, ha, hg, gate)
        xn = np.array(ha)
        log["pre_done"] = time.perf_counter()
        fut = pool.submit(ane, xn)
        yg = gpu(hg)
        mx.async_eval(yg)
        log["gpu_enq"] = time.perf_counter()
        mx.eval(yg)  # blocks until the GPU rows MLP is finished
        log["gpu_done"] = time.perf_counter()
        y, hm = fut.result()
        log["join_done"] = time.perf_counter()
        out = epi(x1, mx.array(np.array(y, copy=True)), mx.array(np.array(hm, copy=True)), yg, gate)
        mx.eval(out)
        log["epi_done"] = time.perf_counter()
        return out, {k: round((v - t0) * 1e3, 1) for k, v in log.items()}

    return run


runs = [layer(i, b, fn, g, x) for i, (b, fn, g) in enumerate(zip(blocks, fns, gains))]
for it in range(3):
    y = x
    for i, r in enumerate(runs):
        y, tl = r(y)
        if it >= 1:  # first pass compiles
            rows.append({"layer": i, **tl})
print(json.dumps(rows, indent=1))
keys = ["pre_done", "ane_start", "gpu_enq", "gpu_done", "ane_end", "join_done", "epi_done"]
print("median ms from layer entry:", {k: float(np.median([r[k] for r in rows])) for k in keys})
