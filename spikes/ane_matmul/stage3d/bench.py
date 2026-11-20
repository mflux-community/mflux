"""Stage 3d benchmark: T1-T4 for the row-split fused MLP. Synthetic random weights at real Z-Image dims
(no downloads, no image generation). Real dependency chain: block N+1 consumes block N's output.

  uv run python spikes/ane_matmul/stage3d/bench.py t1 --backend coremltools
  uv run python spikes/ane_matmul/stage3d/bench.py t2 --backend bridge --shares 0.2 0.25
  uv run python spikes/ane_matmul/stage3d/bench.py t3 --backend bridge --shares 0.25 0.3
  uv run python spikes/ane_matmul/stage3d/bench.py t4 --backend bridge --shares 0.25

--backend mirror runs the numpy fp16 mirror instead of the ANE: validates the harness anywhere, gate-invalid.
Reboot first, hold s3a.LOCK, keep watchdog.sh armed (pass --lock-file to check the lock exists).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mlp_program import DIM, MlpSpec, plan_split, reference_fp16  # noqa: E402
from rowsplit import (  # noqa: E402
    AneBackend,
    BridgeBackend,
    CallStats,
    CompiledSplitBlock,
    CoreMLToolsBackend,
    PrimitiveSplitFFN,
    RowSplitBlock,
    RowSplitFFN,
    prepare_block_weights,
)

SEQ = 4224
N_HEADS = 30
N_BLOCKS = 34
N_REFINERS = 2
SEED = 0


class MirrorBackend(AneBackend):
    weights_copied = True

    def run(self, spec, weights, x):
        y, stats = reference_fp16(x, {"w1": weights.w1, "w3": weights.w3, "w2": weights.w2}, weights.scales, spec)
        return y, stats.get("rowmax")


# ---------- system probes ----------


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001 -- a missing sysctl/vm_stat just yields NaN probes
        return ""


def swap_used_mb() -> float:
    m = re.search(r"used = ([\d.]+)M", sh(["sysctl", "-n", "vm.swapusage"]))
    return float(m.group(1)) if m else float("nan")


def wired_gb() -> float:
    out = sh(["vm_stat"])
    page = re.search(r"page size of (\d+)", out)
    wired = re.search(r"Pages wired down:\s+(\d+)", out)
    return int(wired.group(1)) * int(page.group(1)) / 1e9 if page and wired else float("nan")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


class MacmonSampler:
    def __init__(self):
        self.samples: list[float] = []
        self._proc = None

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(["macmon", "pipe", "-s", "0", "-i", "500"], stdout=subprocess.PIPE, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except FileNotFoundError:
            self._proc = None
        return self

    def _read(self):
        for line in self._proc.stdout:
            try:
                self.samples.append(float(json.loads(line)["ane_power"]))
            except Exception:  # noqa: BLE001 -- macmon emits header and non-power lines too
                pass

    def __exit__(self, *exc):
        if self._proc:
            self._proc.terminate()

    def median_w(self) -> float:
        return float(np.median(self.samples)) if self.samples else float("nan")


def stats(times: list[float]) -> dict:
    a = np.array(times) * 1e3
    return {"median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)), "n": len(a)}


# ---------- synthetic model ----------


def make_blocks(n: int):
    import mlx.core as mx
    from mlx.utils import tree_map

    from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock

    mx.random.seed(SEED)
    blocks = []
    for _ in range(n):
        b = ZImageTransformerBlock(DIM, N_HEADS)
        b.update(tree_map(lambda p: p.astype(mx.bfloat16), b.parameters()))
        mx.eval(b.parameters())
        blocks.append(b)
    return blocks


def make_inputs():
    import mlx.core as mx

    k = mx.random.key(SEED)
    x = mx.random.normal((1, SEQ, DIM), key=k).astype(mx.bfloat16)
    freqs = mx.random.normal((SEQ, 64, 2), key=k)
    t_emb = mx.random.normal((1, 256), key=k).astype(mx.bfloat16)
    mask = mx.ones((1, SEQ), dtype=mx.bool_)
    mx.eval(x, freqs, t_emb, mask)
    return x, mask, freqs, t_emb


def spec_for(args, m: int) -> MlpSpec:
    """One place to build the program spec, so t1/t2/t3/t4 cannot drift apart."""
    return MlpSpec(
        mpad=m,
        weight_dtype=args.weight_dtype,
        scale_mode=args.scale_mode,
        w2_scale=args.w2_scale,
        range_fix=args.range_fix,
    )


def make_backend(name: str, cache: Path, shared_surfaces: bool = False) -> AneBackend:
    return {
        "coremltools": lambda: CoreMLToolsBackend(cache),
        "bridge": lambda: BridgeBackend(cache, shared_surfaces=shared_surfaces),
        "mirror": MirrorBackend,
    }[name]()


def make_split_ffn(mode, ff, backend, spec, sink):
    """threaded = the host-sync path that times the phases; primitive = the form generation uses."""
    w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
    if mode == "primitive":
        if not hasattr(backend, "ext"):
            raise ValueError("--patch-mode primitive needs --backend bridge; mirror/coremltools have no ext")
        model, _outs = backend._prepare(spec)  # caches the program and its output buffers
        return PrimitiveSplitFFN(ff, backend.ext, model, spec, w)
    return RowSplitFFN(ff, backend, spec, w, sink)


def split_blocks(blocks, backend, spec, stats_sink, mode="threaded"):
    out = []
    for b in blocks:
        ff = b.feed_forward
        split = make_split_ffn(mode, ff, backend, spec, stats_sink)
        # threaded wraps (leaves the block alone); primitive swaps per call so the block's own
        # forward stays the thing under test and the gpu_only arm keeps its untouched FFN
        out.append(CompiledSplitBlock(b, split) if mode == "primitive" else RowSplitBlock(b, split))
    return out


# ---------- measured runs ----------


def timed(fn) -> dict:
    """One measured run with the quiet gate: swap growth invalidates it."""
    swap0, wired0 = swap_used_mb(), wired_gb()
    with MacmonSampler() as mm:
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
    swap1, wired1 = swap_used_mb(), wired_gb()
    return {
        "seconds": dt,
        "valid": not (swap1 > swap0 + 1.0),
        "swap_mb": [swap0, swap1],
        "wired_gb": [wired0, wired1],
        "peak_rss_gb": peak_rss_gb(),
        "ane_power_w_median": mm.median_w(),
        "finite": bool(out),
    }


def chain(blocks, x, mask, freqs, t_emb) -> bool:
    import mlx.core as mx

    for b in blocks:
        x = b(x=x, attn_mask=mask, freqs_cis=freqs, t_emb=t_emb)
        mx.eval(x)
    return bool(mx.all(mx.isfinite(x)).item())


def mlp_chain(ffs, norms, x) -> bool:
    import mlx.core as mx

    for ff, norm in zip(ffs, norms):
        x = x + 0.1 * norm(ff(x))
        mx.eval(x)
    return bool(mx.all(mx.isfinite(x)).item())


def interleave(arms: dict, repeats: int) -> dict:
    results = {k: [] for k in arms}
    for _ in range(repeats):
        for name, fn in arms.items():  # ABAB...
            results[name].append(timed(fn))
    summary = {}
    for name, runs in results.items():
        good = [r for r in runs if r["valid"]]
        summary[name] = {
            **(stats([r["seconds"] for r in good]) if good else {}),
            "discarded_swap_growth": len(runs) - len(good),
            "runs": runs,
        }
    return summary


# ---------- tests ----------


def t1(args, backend):
    import mlx.core as mx

    blocks = make_blocks(1)
    ff = blocks[0].feed_forward
    x, *_ = make_inputs()
    rows = []
    for share in args.shares:
        m, _ = plan_split(SEQ, share)
        spec = spec_for(args, m)
        w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
        ref = ff(x[:, :m]).astype(mx.float32)
        y_prog, h_max = backend.run(spec, w, np.array(x[:, :m].astype(mx.float16)))
        y = np.asarray(y_prog, dtype=np.float32)
        if h_max is not None:  # the range fix restores h's row max in fp32, never inside the program
            y = y * np.asarray(h_max, dtype=np.float32)
        y = y * w.gain
        r = np.array(ref).ravel().astype(np.float64)
        yv = y.astype(np.float64).ravel()
        c = float(r @ yv / (np.linalg.norm(r) * np.linalg.norm(yv)))
        # relerr is the scale-sensitive half of the check. The cosine gate on its own scored a
        # 128x gain error at 0.9998; the mirror's intended math sits at relerr ~0.008 vs bf16,
        # so 0.02 admits correct output and rejects any constant-gain mistake.
        relerr = float(np.linalg.norm(yv - r) / np.linalg.norm(r))
        ratio = float(r @ yv / (yv @ yv)) if yv @ yv else float("nan")
        rows.append(
            {
                "share": share,
                "mpad": m,
                "cosine": c,
                "relerr": relerr,
                "gain_ratio": ratio,
                "nonfinite": int((~np.isfinite(y)).sum()),
                "max_abs_y": float(np.nanmax(np.abs(y))),
                "pass": c >= 0.9999 and relerr <= 0.02 and np.isfinite(y).all(),
            }
        )
        print(rows[-1])
    return {"t1": rows, "compute_plan": getattr(backend, "plan_reports", {})}


def t2(args, backend):
    from mlx import nn

    blocks = make_blocks(N_BLOCKS)
    ffs = [b.feed_forward for b in blocks]
    norms = [nn.RMSNorm(DIM) for _ in blocks]
    x, *_ = make_inputs()
    sink = CallStats()
    arms = {"gpu_only": lambda: mlp_chain(ffs, norms, x)}
    for share in args.shares:
        m, _ = plan_split(SEQ, share)
        spec = spec_for(args, m)
        sffs = [make_split_ffn(args.patch_mode, f, backend, spec, sink) for f in ffs]
        arms[f"share_{share}_m{m}"] = lambda s=sffs: mlp_chain(s, norms, x)
    mlp_chain(ffs, norms, x)  # warm-up
    return {"t2": interleave(arms, args.repeats), "per_call_ms": sink.summary_ms()}


def t3_t4(args, backend, ladder):
    blocks = make_blocks(N_BLOCKS + N_REFINERS)
    refiners, main = blocks[:N_REFINERS], blocks[N_REFINERS:]
    x, mask, freqs, t_emb = make_inputs()
    out = {}
    for n in ladder:
        sink = CallStats()
        arms = {"gpu_only": lambda n=n: chain(refiners + main[:n], x, mask, freqs, t_emb)}
        for share in args.shares:
            m, _ = plan_split(SEQ, share)
            spec = spec_for(args, m)
            sb = split_blocks(main[:n], backend, spec, sink, args.patch_mode)
            arms[f"share_{share}_m{m}"] = lambda sb=sb, n=n: chain(refiners + sb, x, mask, freqs, t_emb)
        chain(refiners + main[:n], x, mask, freqs, t_emb)  # warm-up
        out[f"blocks_{n}"] = {"arms": interleave(arms, args.repeats), "per_call_ms": sink.summary_ms()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["t1", "t2", "t3", "t4"])
    ap.add_argument("--backend", choices=["coremltools", "bridge", "mirror"], default="bridge")
    ap.add_argument("--shares", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--weight-dtype", default="int8", choices=["int8", "fp16"])
    ap.add_argument("--scale-mode", default="weight", choices=["weight", "output"])
    ap.add_argument("--w2-scale", default="program", choices=["program", "epilogue"])
    ap.add_argument(
        "--range-fix",
        action="store_true",
        help="the a+b fix from report section 5h: 1/sqrt(K) dequantise constant plus per-row h"
        " normalisation restored in the fp32 epilogue. Needed for real trained weights, whose"
        " partial sums otherwise hit 3.6e7 against fp16's 65504 ceiling.",
    )
    ap.add_argument("--cache", default=str(Path(__file__).parent / "cache"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--lock-wait-s", type=int, default=3600, help="how long to wait for BENCH.lock on timed runs")
    ap.add_argument(
        "--weight-surfaces",
        choices=["per-block", "shared"],
        default="per-block",
        help="per-block wraps each block's own weight buffers (zero copy, but the kernel wires ~118 MB"
        " per block); shared keeps ONE surface set and blits weights in per call (~5.1 ms), which holds"
        " wired memory flat. The blit runs inside the timed region, so shared timings include it.",
    )
    ap.add_argument(
        "--patch-mode",
        choices=["threaded", "primitive"],
        default="threaded",
        help="threaded = the host-sync path, which times the phases and produced the T2-T4 numbers. "
        "primitive = the MLX primitive that generation actually uses, with MLX doing the scheduling. "
        "They are not interchangeable: only primitive can run inside mx.compile.",
    )
    args = ap.parse_args()

    # Timed runs take the cooperative BENCH.lock that stages 3a-3c already use, so two agents
    # cannot contend for the GPU and corrupt each other's numbers. Checking that a path exists
    # is not the same discipline: it passes for a lock somebody else is holding.
    timed = args.test != "t1"
    if timed:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage3a"))
        from s3a import BenchLock

        gate = BenchLock(owner=f"stage3d-{os.getpid()}", wait_s=args.lock_wait_s)
    else:
        gate = contextlib.nullcontext()

    backend = make_backend(args.backend, Path(args.cache), shared_surfaces=args.weight_surfaces == "shared")
    if timed and backend.weights_copied:
        print(
            "WARNING: backend copies weights per call (or is not the ANE): timings are NOT gate-valid", file=sys.stderr
        )

    result = {
        "test": args.test,
        "args": vars(args),
        "weights_copied": backend.weights_copied,
    }
    with gate:
        result["swap_mb_start"] = swap_used_mb()
        result["wired_gb_start"] = wired_gb()
        if args.test == "t1":
            result.update(t1(args, backend))
        elif args.test == "t2":
            result.update(t2(args, backend))
        elif args.test == "t3":
            result.update(t3_t4(args, backend, [N_BLOCKS]))
        else:
            result.update(t3_t4(args, backend, [8, 16, N_BLOCKS]))
        result.update(
            swap_mb_end=swap_used_mb(),
            wired_gb_end=wired_gb(),
            peak_rss_gb=peak_rss_gb(),
            commit=sh(["git", "rev-parse", "--short", "HEAD"]).strip(),
        )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.test}_{args.backend}_{int(time.time())}.json"
    path.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {path}")
    summary = {k: v for k, v in result.items() if k not in ("args",)}
    print(json.dumps(summary, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
