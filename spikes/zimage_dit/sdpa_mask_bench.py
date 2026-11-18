"""SDPA mask-form benchmark for the Z-Image DiT.

Z-Image passes an all-True boolean mask that attention.py:84-88 turns into an additive
float mask of zeros. Mathematically that is a no-op, so the only question is whether it
costs kernel time. This script measures the mask forms in isolation at the real DiT
attention shapes, on both interpreters available on this Mac:

    mflux-upstream/.venv/bin/python  -> mlx 0.32.0 (the pin in pyproject.toml; no force_fused)
    python (system)                  -> mlx 0.32.4 (newest release; has force_fused)

Usage:  python sdpa_mask_bench.py [--reps 50] [--json out.json]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.core.fast import scaled_dot_product_attention

# Z-Image DiT attention geometry: 30 heads x head_dim 128, batch 1.
BATCH = 1
HEADS = 30
HEAD_DIM = 128
DTYPE = mx.bfloat16
# Joint sequence lengths: 512^2 -> 1024 img + 128 cap, 1024^2 -> 4096 + 128, ~1.5MP -> 5824 + 128.
SEQ_LENS = [1152, 4224, 5952]


def _probe_force_fused() -> bool:
    # inspect.signature cannot read kwargs off the nanobind binding, so ask the function itself.
    # head_dim must be one the fused kernel supports, or a ValueError about the shape arrives
    # instead of the TypeError that means "this build has no force_fused kwarg".
    tiny = mx.zeros((1, 2, 8, 128), dtype=DTYPE)
    try:
        mx.eval(scaled_dot_product_attention(tiny, tiny, tiny, scale=1.0, mask=None, force_fused=True))
    except TypeError:
        return False
    return True


SUPPORTS_FORCE_FUSED = _probe_force_fused()


def env_stamp() -> dict:
    """Every result row carries the MLX version and the commit it was produced from.

    A number without both is not reproducible, and this investigation has already shipped one
    claim that a second run contradicted.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "mlx": mx.__version__,
        "commit": commit,
        "commit_short": commit[:8],
        "python": sys.version.split()[0],
        "branch": _branch(),
    }


def _branch() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def attn_macs(seq_len: int, causal: bool = False) -> float:
    # QK^T and A@V: one S*S*D matmul each, per head. A causal mask keeps only the lower triangle,
    # so scoring that row with the dense count reports throughput that was never earned.
    pairs = seq_len * (seq_len + 1) / 2 if causal else float(seq_len * seq_len)
    return 2.0 * pairs * HEAD_DIM * HEADS


def timed(fn, reps: int, warmup: int = 10) -> dict:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn()
        mx.eval(out)
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": sorted(samples)[int(0.9 * (len(samples) - 1))],
        "min_ms": min(samples),
        "reps": reps,
    }


def build_variants(q, k, v, seq_len: int) -> dict:
    """Each entry maps a label to a zero-arg callable; outputs compared against the no-mask run."""
    ones_bool = mx.ones((BATCH, seq_len), dtype=mx.bool_)
    bool_bhsw = mx.ones((BATCH, 1, 1, seq_len), dtype=mx.bool_)
    additive_zeros = mx.zeros((BATCH, 1, 1, seq_len), dtype=DTYPE)

    def sdpa(mask, **kw):
        return lambda: scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5, mask=mask, **kw)

    def mask_built():
        # Exactly attention.py:84-88: bool (B,S) -> additive (B,1,1,S), inside the timed graph.
        mask = mx.where(
            ones_bool[:, None, None, :],
            mx.array(0.0, dtype=q.dtype),
            mx.array(float("-inf"), dtype=q.dtype),
        )
        return scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5, mask=mask)

    variants = {
        "none": sdpa(None),
        "additive_zeros_in": sdpa(additive_zeros),
        "bool_bcast_in": sdpa(bool_bhsw),
        "bool_full_bhsd_in": sdpa(mx.ones((BATCH, HEADS, seq_len, seq_len), dtype=mx.bool_)),
        "mask_built_in_graph": mask_built,
        "causal_str": sdpa("causal"),
    }
    if SUPPORTS_FORCE_FUSED:
        variants["force_fused_none"] = sdpa(None, force_fused=True)
        variants["force_fused_additive"] = sdpa(additive_zeros, force_fused=True)
    return variants


def compare(reference: mx.array, other: mx.array) -> dict:
    a = mx.array(reference.astype(mx.float32))
    b = mx.array(other.astype(mx.float32))
    diff = a - b
    denom = mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b))
    return {
        "cosine": float(mx.sum(a * b) / denom),
        "max_abs_diff": float(mx.max(mx.abs(diff))),
        "bitwise_equal": bool(mx.array_equal(a, b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--json", default=None)
    parser.add_argument("--eager", action="store_true", help="also time without mx.compile")
    args = parser.parse_args()

    stamp = env_stamp()
    results = {**stamp, "force_fused_supported": SUPPORTS_FORCE_FUSED, "runs": []}
    print(
        f"mlx {stamp['mlx']}  python {stamp['python']}  commit {stamp['commit_short']}"
        f" ({stamp['branch']})  force_fused={SUPPORTS_FORCE_FUSED}  dtype={DTYPE}  reps={args.reps}\n"
    )

    for seq_len in SEQ_LENS:
        mx.random.seed(0)
        shape = (BATCH, HEADS, seq_len, HEAD_DIM)
        q = mx.random.normal(shape=shape).astype(DTYPE)
        k = mx.random.normal(shape=shape).astype(DTYPE)
        v = mx.random.normal(shape=shape).astype(DTYPE)
        mx.eval(q, k, v)

        variants = build_variants(q, k, v, seq_len)
        reference = None
        rows = []
        for label, fn in variants.items():
            compiled = mx.compile(fn)
            stats = timed(compiled, reps=args.reps)
            out = mx.array(compiled())
            if reference is None:
                reference = out
            stats.update(
                label=label,
                seq_len=seq_len,
                tflops=attn_macs(seq_len, label == "causal_str") * 2 / (stats["median_ms"] * 1e-3) / 1e12,
            )
            if label != "causal_str":
                stats.update(compare(reference, out))
            if args.eager:
                stats["eager_median_ms"] = timed(fn, reps=max(10, args.reps // 5))["median_ms"]
            rows.append(stats)

        base = next(r for r in rows if r["label"] == "none")
        print(f"--- S={seq_len}  (dense QK^T + AV = {attn_macs(seq_len) * 2 / 1e12:.3f} TFLOP) ---")
        print(f"{'variant':22s} {'median_ms':>10s} {'p90_ms':>9s} {'vs none':>9s} {'TFLOP/s':>8s}  identical")
        for row in sorted(rows, key=lambda r: r["median_ms"]):
            delta = (row["median_ms"] / base["median_ms"] - 1) * 100
            same = (
                "-" if "bitwise_equal" not in row else ("yes" if row["bitwise_equal"] else f"cos={row['cosine']:.6f}")
            )
            print(
                f"{row['label']:22s} {row['median_ms']:10.3f} {row['p90_ms']:9.3f} "
                f"{delta:+8.1f}% {row['tflops']:8.2f}  {same}"
            )
        print()
        results["runs"].extend(rows)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
