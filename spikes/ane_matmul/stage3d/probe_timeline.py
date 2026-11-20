"""Where the ANE call sits inside a block: primitive vs threaded, loop-eval vs mx.compile.

t3's chain calls mx.eval after every block, so it never exercises the compiled form generation
uses. This probe runs both and reads the C++ timeline (wl_timeline) for each ANE call:

  fork_ready     when the ANE's fp16 input became available (the staging cast finished)
  gpu_share_end  when the GPU's rows became available
  primitive      eval_cpu entry, task start, input ready after wait(), predict start, predict end
  threaded       predict start/end only (that path has no eval/task phase)

If the ANE predict runs between fork_ready and gpu_share_end, the two engines overlapped. If it
starts after gpu_share_end, the call was serialised behind the GPU share.

Synthetic weights at real Z-Image dims, like bench.py. No checkpoint, no image.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx  # noqa: E402
from bench import make_backend, make_blocks, make_inputs, spec_for  # noqa: E402
from mlp_program import DIM, plan_split  # noqa: E402
from rowsplit import (  # noqa: E402
    CallStats,
    CompiledSplitBlock,
    PrimitiveSplitFFN,
    RowSplitFFN,
    prepare_block_weights,
)


class MarkedSplitFFN(PrimitiveSplitFFN):
    """The production graph plus two availability marks. Checked against the unmarked form below."""

    def __call__(self, h):
        m = self.spec.mpad
        h_ane = h[:, :m].astype(mx.float16)
        h_gpu = mx.depends([h[:, m:]], [h_ane])[0]
        fork_tok = self.ext.wl_mark("fork_ready", h_ane)
        y, h_max = self.ext.wl_fused(h_ane, *self._mxw, self.model)
        y_gpu = self.ff(h_gpu)
        gpu_tok = self.ext.wl_mark("gpu_share_end", y_gpu)
        y_ane = y.astype(mx.float32) * h_max.astype(mx.float32) * self._gain
        out = mx.concatenate([y_ane.astype(h.dtype), y_gpu.astype(h.dtype)], axis=1)
        # The marks only need to be evaluated, not to gate anything, so they hang off the output.
        return mx.depends([out], [fork_tok, gpu_tok])[0]


def run_chain(blocks, inputs, compiled):
    x, mask, freqs, t_emb = inputs

    def step(x):
        for b in blocks:
            x = b(x=x, attn_mask=mask, freqs_cis=freqs, t_emb=t_emb)
        return x

    if compiled:
        fn = mx.compile(step)
        mx.eval(fn(x))  # warm-up, also builds the compile cache
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(x)
        mx.eval(out)
        dt = time.perf_counter() - t0
    else:
        for b in blocks:  # warm-up
            x = b(x=x, attn_mask=mask, freqs_cis=freqs, t_emb=t_emb)
            mx.eval(x)
        mx.synchronize()
        t0 = time.perf_counter()
        for b in blocks:
            x = b(x=x, attn_mask=mask, freqs_cis=freqs, t_emb=t_emb)
            mx.eval(x)
        dt = time.perf_counter() - t0
        out = x
    return dt, bool(mx.all(mx.isfinite(out)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the result file, so env-var settings do not overwrite")
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--cache", default=str(Path(__file__).parent / "cache"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    from bench import SEQ

    m, _ = plan_split(SEQ, args.share)
    spec = spec_for(argparse.Namespace(weight_dtype="int8", scale_mode="output", w2_scale="program", range_fix=True), m)
    backend = make_backend("bridge", Path(args.cache))
    model, _outs = backend._prepare(spec)
    ext = backend.ext

    blocks = make_blocks(args.blocks)
    inputs = make_inputs()
    env = {k: os.environ.get(k, "-") for k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_METAL_FAST_SYNCH", "MLX_BFS_MAX_WIDTH")}
    print(f"blocks={args.blocks} S={SEQ} mpad={m} spec={spec.name} env={env}", flush=True)

    # The marked graph must compute the same thing as the production one.
    ff0 = blocks[0].feed_forward
    w0 = prepare_block_weights(spec, ff0.w1.weight, ff0.w3.weight, ff0.w2.weight)
    h = mx.random.normal((1, SEQ, DIM)).astype(mx.bfloat16)
    plain = PrimitiveSplitFFN(ff0, ext, model, spec, w0)(h)
    marked = MarkedSplitFFN(ff0, ext, model, spec, w0)(h)
    mx.eval(plain, marked)
    d = float(mx.max(mx.abs(plain.astype(mx.float32) - marked.astype(mx.float32))).item())
    print(f"marked vs production graph: maxdiff {d:.6f}", flush=True)
    assert d == 0.0, "the probe's graph drifted from the production form"

    results = {}
    for name, mode, compiled in [
        ("gpu_only_loop", None, False),
        ("primitive_loop", "primitive", False),
        ("primitive_compiled", "primitive", True),
        ("threaded_loop", "threaded", False),
    ]:
        wrapped = []
        stats = CallStats()
        for b in blocks:
            if mode is None:
                wrapped.append(b)
                continue
            ff = b.feed_forward
            w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
            split = (
                MarkedSplitFFN(ff, ext, model, spec, w)
                if mode == "primitive"
                else RowSplitFFN(ff, backend, spec, w, stats)
            )
            wrapped.append(CompiledSplitBlock(b, split))
        ext.wl_timeline_reset()
        dt, finite = run_chain(wrapped, inputs, compiled)
        tl = ext.wl_timeline()
        results[name] = {"seconds": dt, "finite": finite, "timeline": tl, "per_call_ms": stats.summary_ms()}
        print(f"{name:20s} {dt * 1e3:8.1f} ms  finite={finite}  timeline rows={len(tl)}", flush=True)
        if mode == "threaded":
            print(f"{'':20s} per-call {stats.summary_ms()}", flush=True)

    # Pair the marks with the ANE calls in block order and print the overlap picture.
    for name in ("primitive_loop", "primitive_compiled"):
        tl = results[name]["timeline"]
        prim = [r for r in tl if r[0] == "primitive"]
        fork = [r for r in tl if r[0] == "fork_ready"]
        gpu = [r for r in tl if r[0] == "gpu_share_end"]
        print(f"\n=== {name}: {len(prim)} ANE calls, {len(fork)} fork marks, {len(gpu)} gpu marks ===")
        print(
            " blk | fork_ready | gpu_end | eval_cpu | task_start | in_ready | predict_start | predict_end | predict_ms | ane_inside_gpu"
        )
        t0 = min(r[1] for r in tl if r[1] >= 0)
        for i, p in enumerate(prim):
            f = fork[i][3] - t0 if i < len(fork) else float("nan")
            g = gpu[i][3] - t0 if i < len(gpu) else float("nan")
            inside = "YES" if (f <= p[4] and p[5] <= g) else ("late" if p[4] > g else "early")
            print(
                f"{i:4d} | {f:10.1f} | {g:7.1f} | {p[1] - t0:8.1f} | {p[2] - t0:10.1f} | {p[3] - t0:8.1f} | "
                f"{p[4] - t0:13.1f} | {p[5] - t0:11.1f} | {p[5] - p[4]:10.1f} | {inside}"
            )
    th = [r for r in results["threaded_loop"]["timeline"] if r[0] == "threaded"]
    if th:
        durs = np.array([r[5] - r[4] for r in th])
        print(
            f"\nthreaded predict ms: median {np.median(durs):.1f} min {durs.min():.1f} max {durs.max():.1f} n={len(durs)}"
        )
    for name in ("primitive_loop", "primitive_compiled"):
        pr = [r for r in results[name]["timeline"] if r[0] == "primitive"]
        if pr:
            durs = np.array([r[5] - r[4] for r in pr])
            waits = np.array([r[3] - r[2] for r in pr])
            print(
                f"{name} predict ms: median {np.median(durs):.1f} max {durs.max():.1f} | input wait ms median {np.median(waits):.1f} max {waits.max():.1f}"
            )

    for name in ("primitive_loop", "primitive_compiled"):
        pr = [r for r in results[name]["timeline"] if r[0] == "primitive"]
        if len(pr) > 1:
            half = pr[len(pr) // 2 :]  # the timed half, after warm-up
            gap = np.median([r[2] - r[1] for r in half])
            wt = np.median([r[3] - r[2] for r in half])
            pd = np.median([r[5] - r[4] for r in half])
            per = np.median([half[i + 1][1] - half[i][1] for i in range(len(half) - 1)])
            print(
                f"SUMMARY {name:19s} eval->task {gap:6.1f} ms | wait {wt:5.1f} | predict {pd:5.1f} | period {per:6.1f}"
            )
    path = Path(args.out) / f"timeline_{args.blocks}b{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps({"spec": spec.name, "blocks": args.blocks, "env": env, "results": results}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
