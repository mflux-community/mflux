"""Eager orchestration test: per-block compile + threaded ANE MLP between compiled calls.

Question: the row-split MLP gave ~1.2x with the ANE call on a Python thread, but a thread cannot
live inside whole-step mx.compile, and the MLX primitive form starts the ANE ~96 ms late (~1.0x).
Can each block be compiled on its own, with plain Python launching the threaded ANE MLP between
the compiled calls, and lose little?

Chain (real Z-Image Turbo bf16 weights, S=4224 at 1024x1024): 2 noise_refiner blocks + 30 main
layers. The refiners stay on the GPU in every arm. The real model has 30 main layers; the old
synthetic T4 ladder used 34, so absolute times are not comparable with T4, only the ratios are.

Arms (speed ratios are  time_of_first / time_of_second, so >1 means the second arm is faster):
  A  GPU only, whole chain under one mx.compile (today's mflux shape)
  B  GPU only, one mx.compile per block, plain Python loop (optional mx.async_eval per block)
  C  B + threaded ANE MLP per main layer:
       compiled pre (attention + fork) -> async_eval -> ANE on worker thread ->
       compiled GPU-rows MLP -> join -> compiled epilogue
  D  C with MLX_METAL_FAST_SYNCH=1 (must be a separate process: run with --arms C and set the env)

  .venv/bin/python spikes/ane_matmul/stage3d/eager_bench.py --repeats 5           # real, needs BENCH.lock
  .venv/bin/python spikes/ane_matmul/stage3d/eager_bench.py --synthetic --fake-ane --no-lock  # weight-free
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
from bench import interleave, peak_rss_gb, sh, swap_used_mb, wired_gb  # noqa: E402
from mlp_program import DIM, MlpSpec, plan_split  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock  # noqa: E402

N_HEADS = 30
SEED = 0
REPO = Path.home() / ".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo"


# ---------- model and inputs ----------


def load_real_blocks(n_refiners: int, n_main: int):
    """Real bf16 weights straight from the HF snapshot. Only the transformer blocks are loaded."""
    snap = next(REPO.glob("snapshots/*")).resolve() / "transformer"
    wmap = json.loads((snap / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    names = [f"noise_refiner.{i}" for i in range(n_refiners)] + [f"layers.{i}" for i in range(n_main)]
    blocks = {n: ZImageTransformerBlock(DIM, N_HEADS) for n in names}
    for shard in sorted({v for k, v in wmap.items() if ".".join(k.split(".")[:2]) in blocks}):
        weights = mx.load(str(snap / shard))
        for name, blk in blocks.items():
            pairs = [
                (k[len(name) + 1 :], v.astype(mx.bfloat16)) for k, v in weights.items() if k.startswith(name + ".")
            ]
            if pairs:
                blk.load_weights(pairs, strict=False)
        del weights
    for blk in blocks.values():
        mx.eval(blk.parameters())
    ordered = [blocks[n] for n in names]
    return ordered[:n_refiners], ordered[n_refiners:]


def synthetic_blocks(n_refiners: int, n_main: int, dim: int, heads: int):
    from mlx.utils import tree_map

    mx.random.seed(SEED)
    out = []
    for _ in range(n_refiners + n_main):
        b = ZImageTransformerBlock(dim, heads)
        b.update(tree_map(lambda p: p.astype(mx.bfloat16), b.parameters()))
        mx.eval(b.parameters())
        out.append(b)
    return out[:n_refiners], out[n_refiners:]


def make_inputs(seq: int, dim: int):
    k = mx.random.key(SEED)
    x = mx.random.normal((1, seq, dim), key=k).astype(mx.bfloat16)
    angles = mx.random.uniform(-3.14, 3.14, (seq, 64), key=k)
    freqs = mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)
    t_emb = mx.random.normal((1, min(dim, 256)), key=k).astype(mx.bfloat16)
    mask = mx.ones((1, seq), dtype=mx.bool_)
    mx.eval(x, freqs, t_emb, mask)
    return x, mask, freqs, t_emb


# ---------- the three compiled pieces of a split block ----------


def make_pre(block, m):
    """Everything up to the MLP. Returns the residual stream, the ANE rows (fp16), the GPU rows, the gate."""

    def pre(x, mask, freqs, t_emb):
        modulation = mx.expand_dims(block.adaLN_modulation[0](t_emb), axis=1)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation, 4, axis=2)
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        gate_msa, gate_mlp = mx.tanh(gate_msa), mx.tanh(gate_mlp)
        attn_out = block.attention(block.attention_norm1(x) * scale_msa, attention_mask=mask, freqs_cis=freqs)
        x1 = x + gate_msa * block.attention_norm2(attn_out)
        h = block.ffn_norm1(x1) * scale_mlp
        return x1, h[:, :m].astype(mx.float16), h[:, m:], gate_mlp

    return mx.compile(pre)


def make_gpu_mlp(block):
    return mx.compile(lambda h: block.feed_forward(h))


def make_epilogue(block, gain):
    limit = float(mx.finfo(mx.float16).max)

    def epilogue(x1, y_ane, h_max, y_gpu, gate_mlp):
        # the range fix restores h's row max in fp32, outside the program (restoring it in fp16 overflows)
        y = y_ane.astype(mx.float32) * h_max.astype(mx.float32) * gain
        y = mx.clip(y, -limit, limit).astype(x1.dtype)
        ffn_out = mx.concatenate([y, y_gpu.astype(x1.dtype)], axis=1)
        return x1 + gate_mlp * block.ffn_norm2(ffn_out)

    return mx.compile(epilogue)


class SplitBlock:
    """One main layer in arm C. Plain Python between three compiled calls; the ANE call is on a thread."""

    def __init__(self, block, ane_run, gain, m, pool, phases):
        self.pre = make_pre(block, m)
        self.gpu_mlp = make_gpu_mlp(block)
        self.epilogue = make_epilogue(block, mx.array(gain))
        self.ane_run, self.pool, self.phases = ane_run, pool, phases

    def __call__(self, x, mask, freqs, t_emb):
        t0 = time.perf_counter()
        x1, h_ane, h_gpu, gate = self.pre(x, mask, freqs, t_emb)
        mx.async_eval(x1, h_ane, h_gpu, gate)
        x_np = np.array(h_ane)  # waits for the pre graph only
        t1 = time.perf_counter()
        fut = self.pool.submit(self.ane_run, x_np)
        y_gpu = self.gpu_mlp(h_gpu)
        mx.async_eval(y_gpu)
        t2 = time.perf_counter()
        y_np, h_max = fut.result()
        t3 = time.perf_counter()
        # copy: the backend reuses its output buffers, and the epilogue is lazy
        y_ane = mx.array(np.array(y_np, dtype=np.float16, copy=True))
        hm = mx.array(np.array(h_max, dtype=np.float16, copy=True))
        out = self.epilogue(x1, y_ane, hm, y_gpu, gate)
        self.phases["pre_wait"].append(t1 - t0)
        self.phases["gpu_enqueue"].append(t2 - t1)
        self.phases["join_wait"].append(t3 - t2)
        return out


# ---------- ANE backends ----------


class FakeAne:
    """Weight-free stand-in: computes the MLP in numpy float32 for a block, returns it as the ANE would."""

    def __init__(self, ff, delay_s=0.0):
        self.w = [np.array(w.astype(mx.float32)) for w in (ff.w1.weight, ff.w3.weight, ff.w2.weight)]
        self.delay_s = delay_s

    def __call__(self, x):
        time.sleep(self.delay_s)
        x = x.astype(np.float32)[0]
        a, b = x @ self.w[0].T, x @ self.w[1].T
        h = a / (1 + np.exp(-a)) * b
        y = (h @ self.w[2].T)[None]
        return y.astype(np.float16), np.ones((1, x.shape[0], 1), dtype=np.float16)


def real_ane_fns(blocks, spec, cache):
    from rowsplit import BridgeBackend, prepare_block_weights

    backend = BridgeBackend(cache)
    fns, gains = [], []
    for b in blocks:
        ff = b.feed_forward
        w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
        fns.append(lambda x, w=w: backend.run(spec, w, x))
        gains.append(w.gain)
    return fns, gains, backend


# ---------- arms ----------


def run_arm_a(step_fn, x, mask, freqs, t_emb):
    out = step_fn(x, mask, freqs, t_emb)
    mx.eval(out)
    return bool(mx.all(mx.isfinite(out)).item())


def chain_python(callables, x, mask, freqs, t_emb, async_between=False):
    for c in callables:
        x = c(x, mask, freqs, t_emb)
        if async_between:
            mx.async_eval(x)
    mx.eval(x)
    return x


def final_output(callables, x, mask, freqs, t_emb):
    return np.array(chain_python(callables, x, mask, freqs, t_emb).astype(mx.float32)).astype(np.float64).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C", help="comma list of A,B,C; D = run --arms C with MLX_METAL_FAST_SYNCH=1")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--share", type=float, default=0.50)
    ap.add_argument("--seq", type=int, default=4224)
    ap.add_argument("--refiners", type=int, default=2)
    ap.add_argument("--main", type=int, default=30)
    ap.add_argument("--b-async-eval", action="store_true", help="mx.async_eval between blocks in arm B")
    ap.add_argument("--synthetic", action="store_true", help="random weights, small dims: weight-free smoke test")
    ap.add_argument("--fake-ane", action="store_true", help="numpy stand-in for the ANE (implies --synthetic dims)")
    ap.add_argument("--fake-delay-ms", type=float, default=0.0)
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--lock-wait-s", type=int, default=3600)
    ap.add_argument("--cache", default=str(HERE / "cache"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--label", default="eager")
    args = ap.parse_args()
    arms = args.arms.split(",")

    dim, heads = (256, 2) if args.fake_ane else (DIM, N_HEADS)
    seq = 768 if args.fake_ane else args.seq
    m = 256 if args.fake_ane else plan_split(seq, args.share)[0]

    gate = None
    if not args.no_lock:
        sys.path.insert(0, str(HERE.parent / "stage3a"))
        from s3a import BenchLock

        gate = BenchLock(owner=f"eager-{os.getpid()}", wait_s=args.lock_wait_s)
        gate.__enter__()
    try:
        result = {
            "args": vars(args),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "fast_synch": os.environ.get("MLX_METAL_FAST_SYNCH"),
            "swap_mb_start": swap_used_mb(),
            "wired_gb_start": wired_gb(),
        }
        t_load = time.perf_counter()
        if args.synthetic or args.fake_ane:
            refiners, main_blocks = synthetic_blocks(args.refiners, args.main, dim, heads)
        else:
            refiners, main_blocks = load_real_blocks(args.refiners, args.main)
        x, mask, freqs, t_emb = make_inputs(seq, dim)
        result["load_s"] = time.perf_counter() - t_load
        result["swap_mb_after_load"] = swap_used_mb()
        result["wired_gb_after_load"] = wired_gb()

        phases = {"pre_wait": [], "gpu_enqueue": [], "join_wait": []}
        pool = ThreadPoolExecutor(max_workers=1)
        blocks_all = refiners + main_blocks
        callables = {}
        backend = None
        if "A" in arms:

            def whole(x, mask, freqs, t_emb):
                for b in blocks_all:
                    x = b(x, mask, freqs, t_emb)
                return x

            callables["A"] = mx.compile(whole)
        if "B" in arms:
            callables["B"] = [
                mx.compile(lambda x, mask, freqs, t_emb, b=b: b(x, mask, freqs, t_emb)) for b in blocks_all
            ]
        if "C" in arms:
            if args.fake_ane:
                fns = [FakeAne(b.feed_forward, args.fake_delay_ms / 1e3) for b in main_blocks]
                gains = [np.ones((dim,), dtype=np.float32)] * len(main_blocks)
            else:
                spec = MlpSpec(mpad=m, scale_mode="output", range_fix=True)
                fns, gains, backend = real_ane_fns(main_blocks, spec, Path(args.cache))
            split = [SplitBlock(b, fn, g, m, pool, phases) for b, fn, g in zip(main_blocks, fns, gains)]
            callables["C"] = [
                mx.compile(lambda x, mask, freqs, t_emb, b=b: b(x, mask, freqs, t_emb)) for b in refiners
            ] + split

        def arm_fn(name):
            if name == "A":
                return lambda: run_arm_a(callables["A"], x, mask, freqs, t_emb)
            async_b = args.b_async_eval and name == "B"

            def f():
                out = chain_python(callables[name], x, mask, freqs, t_emb, async_between=async_b)
                return bool(mx.all(mx.isfinite(out)).item())

            return f

        fns_by_arm = {a: arm_fn(a) for a in arms}
        for a in arms:  # warm-up: compiles and primes every cache
            fns_by_arm[a]()
        for k in phases:
            phases[k].clear()
        result["swap_mb_after_warmup"] = swap_used_mb()

        result["arms"] = interleave(fns_by_arm, args.repeats)
        for a in arms:
            med = result["arms"][a].get("median_ms")
            print(f"arm {a}: median {med} ms discarded={result['arms'][a]['discarded_swap_growth']}", flush=True)
        if "C" in arms:
            result["c_phase_ms_median"] = {k: 1e3 * float(np.median(v)) for k, v in phases.items() if v}
            result["c_ane_calls"] = len(phases["join_wait"])

        # per-block time split: a profile pass that syncs after every block (not the timed arms)
        if "B" in arms:
            tb = []
            xx = x
            for c in callables["B"]:
                t0 = time.perf_counter()
                xx = c(xx, mask, freqs, t_emb)
                mx.eval(xx)
                tb.append(time.perf_counter() - t0)
            result["profile_B_block_ms"] = [round(1e3 * t, 2) for t in tb]
        if "C" in arms:
            for k in phases:
                phases[k].clear()
            tc = []
            xx = x
            for c in callables["C"]:
                t0 = time.perf_counter()
                xx = c(xx, mask, freqs, t_emb)
                mx.eval(xx)
                tc.append(time.perf_counter() - t0)
            result["profile_C_block_ms"] = [round(1e3 * t, 2) for t in tc]
            result["profile_C_phase_ms"] = {k: [round(1e3 * t, 2) for t in v] for k, v in phases.items()}

        # correctness: final latent of each arm against A
        if "A" in arms:
            ref = np.array(callables["A"](x, mask, freqs, t_emb).astype(mx.float32)).astype(np.float64).ravel()
            result["accuracy_vs_A"] = {}
            for a in arms:
                if a == "A":
                    continue
                got = final_output(callables[a], x, mask, freqs, t_emb)
                cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
                rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
                result["accuracy_vs_A"][a] = {"cosine": cos, "rel_l2": rel, "nonfinite": int((~np.isfinite(got)).sum())}

        # ratios
        med = {a: result["arms"][a]["median_ms"] for a in arms if result["arms"][a].get("median_ms")}
        ratios = {}
        if "A" in med and "B" in med:
            ratios["B/A"] = med["A"] / med["B"]
        if "A" in med and "C" in med:
            ratios["C/A"] = med["A"] / med["C"]
        if "B" in med and "C" in med:
            ratios["C/B"] = med["B"] / med["C"]
        result["speed_ratios"] = ratios
        result.update(
            swap_mb_end=swap_used_mb(),
            wired_gb_end=wired_gb(),
            peak_rss_gb=peak_rss_gb(),
            commit=sh(["git", "rev-parse", "--short", "HEAD"]).strip(),
        )
        if backend is not None and hasattr(backend, "plan_reports"):
            result["plan_report"] = backend.plan_reports
    finally:
        if gate is not None:
            gate.__exit__(None, None, None)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.label}_{int(time.time())}.json"
    path.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {path}")
    keep = {k: v for k, v in result.items() if k not in ("args",)}
    for a in keep.get("arms", {}):
        keep["arms"][a] = {k: v for k, v in keep["arms"][a].items() if k != "runs"}
    print(json.dumps(keep, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
