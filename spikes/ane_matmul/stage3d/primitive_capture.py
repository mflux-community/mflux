"""Metal capture of the OLD blocking-primitive path (compiled, 2 main layers, after warm-up) plus the
in-primitive timestamps (eval_cpu entry, task start, input ready, predict start/end) from the same run.

  MTL_CAPTURE_ENABLED=1 python primitive_capture.py OUT.gputrace
Evidence for an MLX issue: the CPU-stream task for the ANE call starts late. No Xcode analysis here.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import argparse  # noqa: E402

import mlx.core as mx  # noqa: E402
from bench import SEQ, make_backend, make_blocks, make_inputs, spec_for  # noqa: E402
from mlp_program import plan_split  # noqa: E402
from probe_timeline import MarkedSplitFFN  # noqa: E402
from rowsplit import CompiledSplitBlock, prepare_block_weights  # noqa: E402

out = sys.argv[1]
m, _ = plan_split(SEQ, 0.5)
spec = spec_for(argparse.Namespace(weight_dtype="int8", scale_mode="output", w2_scale="program", range_fix=True), m)
backend = make_backend("bridge", HERE / "cache")
model, _ = backend._prepare(spec)
ext = backend.ext
blocks = make_blocks(2)
x, mask, freqs, t_emb = make_inputs()
wrapped = []
for b in blocks:
    ff = b.feed_forward
    w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
    wrapped.append(CompiledSplitBlock(b, MarkedSplitFFN(ff, ext, model, spec, w)))


def step(x):
    for b in wrapped:
        x = b(x=x, attn_mask=mask, freqs_cis=freqs, t_emb=t_emb)
    return x


fn = mx.compile(step)
for _ in range(2):  # warm-up: compile cache, Core ML load
    mx.eval(fn(x))
mx.synchronize()
ext.wl_timeline_reset()
wall_start = time.time()
perf_start = time.perf_counter()
mx.metal.start_capture(out)
mx.eval(fn(x))
mx.synchronize()
mx.metal.stop_capture()
perf_end = time.perf_counter()
tl = ext.wl_timeline()
prim = [r for r in tl if r[0] == "primitive"]
t0 = min(r[1] for r in tl if r[1] >= 0)
print(
    "capture",
    out,
    "wall_start_epoch",
    wall_start,
    "perf_counter_start",
    perf_start,
    "capture_span_ms",
    (perf_end - perf_start) * 1e3,
)
print("raw timeline rows (tag, eval_cpu, task_start, in_ready, predict_start, predict_end; absolute, native clock):")
print(json.dumps(tl))
print("ANE calls:", len(prim), "| ms from the first timeline mark (t0):")
for i, p in enumerate(prim):
    print(
        f"call {i}: eval_cpu {p[1] - t0:.1f} task_start {p[2] - t0:.1f} (late by {p[2] - p[1]:.1f}) in_ready {p[3] - t0:.1f} predict {p[4] - t0:.1f}..{p[5] - t0:.1f} ({p[5] - p[4]:.1f} ms)"
    )
(Path(out).parent / "primitive_timeline.json").write_text(
    json.dumps({"wall_start_epoch": wall_start, "perf_counter_start": perf_start, "rows": tl, "t0": t0})
)
