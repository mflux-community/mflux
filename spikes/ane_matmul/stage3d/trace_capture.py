"""Programmatic Metal capture of arm A or C over 2 layers (no Instruments / DTServiceHub needed).
MTL_CAPTURE_ENABLED=1 python trace_capture.py ARM OUT.gputrace
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlx.core as mx  # noqa: E402
from eager_bench import SplitBlock, load_real_blocks, make_inputs, real_ane_fns  # noqa: E402
from mlp_program import MlpSpec, plan_split  # noqa: E402

arm, out = sys.argv[1], sys.argv[2]


def _run_blocks(blocks, x, mk, f, t):
    for b in blocks:
        x = b(x, mk, f, t)
    return x


_, blocks = load_real_blocks(0, 4)
x, mask, freqs, t = make_inputs(4224, 3840)
m = plan_split(4224, 0.5)[0]
if arm == "A":
    chain = mx.compile(lambda x, mk, f, t: _run_blocks(blocks[:2], x, mk, f, t))  # the same 2 blocks as arm C
    calls = [lambda x, mk, f, t: chain(x, mk, f, t)]
else:
    spec = MlpSpec(mpad=m, scale_mode="output", range_fix=True)
    fns, gains, _ = real_ane_fns(blocks, spec, HERE / "cache")
    ph = {"pre_wait": [], "gpu_enqueue": [], "join_wait": []}
    pool = ThreadPoolExecutor(max_workers=1)
    calls = [SplitBlock(b, fn, g, m, pool, ph) for b, fn, g in zip(blocks, fns, gains)]
for _ in range(2):  # warm-up
    y = x
    for c in calls:
        y = c(y, mask, freqs, t)
    mx.eval(y)
mx.metal.start_capture(out)
y = x
for c in calls if arm == "A" else calls[:2]:
    y = c(y, mask, freqs, t)
mx.eval(y)
mx.metal.stop_capture()
print("captured", out)
