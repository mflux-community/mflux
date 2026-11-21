"""Linear-layer share of one Z-Image DiT block, at real shapes with synthetic weights.

The Apple Neural Engine offload projection needs one number: what fraction of block time is the linear
layers. This times a real ZImageTransformerBlock (imported from mflux, not a copy) against
the same block stripped to its seven fat matmuls, plus the SDPA call in isolation, for both
bf16 and q8 weights. Everything is mx.compile'd, because production compiles the whole predict.

Activations are passed as traced arguments, never captured as closure constants, so MLX
cannot fold the measurement away. Weights stay captured, which is what production does.

    mflux-upstream/.venv/bin/python block_linear_share.py --reps 15 --bits 16,8
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.core.fast import scaled_dot_product_attention
from mlx.utils import tree_flatten

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock

DIM = 3840
HEADS = 30
HEAD_DIM = DIM // HEADS
HIDDEN = 10240
T_EMB = min(DIM, 256)
CAP_TOKENS = 128
# (height_tokens, width_tokens) after the 8x VAE and the 2x2 patchify, i.e. pixels / 16.
GRIDS = {"512x512": (32, 32), "1024x1024": (64, 64), "1024x1456 (~1.5MP)": (64, 91)}
N_BLOCKS = 34  # 30 main + 2 noise refiner + 2 context refiner
DT = ModelConfig.precision


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


def param_bytes(module: nn.Module) -> int:
    total = 0
    for _, value in tree_flatten(module.parameters()):
        if isinstance(value, mx.array):
            total += value.nbytes
            continue
        for attr in ("weight", "scale", "bias", "scales", "biases"):
            item = getattr(value, attr, None)
            if isinstance(item, mx.array):
                total += item.nbytes
    return total


def timed_interleaved(specs: dict, reps: int, warmup: int = 5) -> dict:
    """Time every variant once per repetition, flipping the variant order on every other repetition.

    Timing each variant in one block lets clock, thermal and cache drift favour whichever variant
    runs first or last. specs maps a label to (fn, args); the raw per-repetition samples in
    milliseconds are returned so the spread can be checked, not only the median.
    """
    for fn, args in specs.values():
        for _ in range(warmup):
            mx.eval(fn(*args))
    labels = list(specs)
    samples = {label: [] for label in labels}
    for rep in range(reps):
        for label in labels if rep % 2 == 0 else labels[::-1]:
            fn, args = specs[label]
            start = time.perf_counter()
            mx.eval(fn(*args))
            samples[label].append((time.perf_counter() - start) * 1e3)
    return samples


def make_inputs(h_tokens: int, w_tokens: int):
    """Joint (image + caption) sequence, laid out the way transformer.py does."""
    seq = h_tokens * w_tokens + CAP_TOKENS
    mx.random.seed(0)
    x = mx.random.normal((1, seq, DIM)).astype(DT) * 0.5
    x_ids = ZImageTransformer._create_coord_grid((1, h_tokens, w_tokens), (CAP_TOKENS + 1, 0, 0)).reshape(-1, 3)
    cap_ids = ZImageTransformer._create_coord_grid((CAP_TOKENS, 1, 1), (1, 0, 0)).reshape(-1, 3)
    rope = RopeEmbedder(theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1024, 512, 512])
    freqs = mx.concatenate([rope(x_ids), rope(cap_ids)], axis=0)
    mask = mx.ones((1, seq), dtype=mx.bool_)  # transformer.py:108/124/139 are always all-True
    t_emb = mx.random.normal((1, T_EMB)).astype(DT)
    mx.eval(x, freqs, mask, t_emb)
    return seq, x, mask, freqs, t_emb


def linear_only(block: ZImageTransformerBlock, x, t_emb):
    """The block's seven fat matmuls plus the adaLN matvec, nothing else."""
    a = block.attention
    f = block.feed_forward
    mod = block.adaLN_modulation[0](t_emb)
    q = a.to_q(x)
    k = a.to_k(x)
    v = a.to_v(x)
    o = a.to_out[0](x)
    h2 = f.w2(nn.silu(f.w1(x)) * f.w3(x))
    return mod, q, k, v, o, h2


def measure(block, label, h_tok, w_tok, reps):
    seq, x, mask, freqs, t_emb = make_inputs(h_tok, w_tok)
    mx.random.seed(1)
    shape = (1, HEADS, seq, HEAD_DIM)
    q = mx.random.normal(shape=shape).astype(DT)
    k = mx.random.normal(shape=shape).astype(DT)
    v = mx.random.normal(shape=shape).astype(DT)
    mx.eval(q, k, v)

    # Distinct signatures, so MLX cannot share a trace between the two mask forms.
    def _full(x_, f_, t_, m_):
        return block(x=x_, attn_mask=m_, freqs_cis=f_, t_emb=t_)

    def _nomask(x_, f_, t_):
        return block(x=x_, attn_mask=None, freqs_cis=f_, t_emb=t_)

    def _sdpa_masked(q_, k_, v_, m_):
        additive = mx.where(m_[:, None, None, :], mx.array(0.0, dtype=DT), mx.array(float("-inf"), dtype=DT))
        return scaled_dot_product_attention(q_, k_, v_, scale=HEAD_DIM**-0.5, mask=additive)

    def _sdpa_none(q_, k_, v_):
        return scaled_dot_product_attention(q_, k_, v_, scale=HEAD_DIM**-0.5, mask=None)

    full = mx.compile(_full)
    nomask = mx.compile(_nomask)
    lin = mx.compile(lambda x_, t_: linear_only(block, x_, t_))
    sdpa_masked = mx.compile(_sdpa_masked)
    sdpa_none = mx.compile(_sdpa_none)

    raw = timed_interleaved(
        {
            "block": (full, (x, freqs, t_emb, mask)),
            "block_nomask": (nomask, (x, freqs, t_emb)),
            "linear": (lin, (x, t_emb)),
            "sdpa_masked": (sdpa_masked, (q, k, v, mask)),
            "sdpa_unmasked": (sdpa_none, (q, k, v)),
        },
        reps,
    )
    t_full, t_nomask, t_lin, t_attn_m, t_attn_n = (
        statistics.median(raw[name]) for name in ("block", "block_nomask", "linear", "sdpa_masked", "sdpa_unmasked")
    )
    other = t_full - t_lin - t_attn_m
    # The seven fat projections run once per token. The adaLN projection consumes t_emb, which is
    # (1, T_EMB), so it runs once per step whatever seq is. Counting it per token overstates the
    # linear FLOPs by about 2.2%.
    token_macs = 4 * DIM * DIM + 3 * DIM * HIDDEN
    linear_flops = 2 * (token_macs * seq + T_EMB * 4 * DIM)
    attn_flops = 4 * seq * seq * DIM

    return {
        "res": label,
        "seq": seq,
        "block_ms": t_full,
        "block_nomask_ms": t_nomask,
        "linear_ms": t_lin,
        "sdpa_masked_ms": t_attn_m,
        "sdpa_unmasked_ms": t_attn_n,
        "other_ms": other,
        "linear_share_pct": 100 * t_lin / t_full,
        "sdpa_share_pct": 100 * t_attn_m / t_full,
        "other_share_pct": 100 * other / t_full,
        "mask_cost_in_block_pct": 100 * (t_full / t_nomask - 1),
        "linear_tflops": linear_flops / (t_lin * 1e-3) / 1e12,
        "sdpa_tflops": attn_flops / (t_attn_m * 1e-3) / 1e12,
        "linear_tflop_per_block": linear_flops / 1e12,
        "sdpa_tflop_per_block": attn_flops / 1e12,
        "step_s_34_blocks": t_full * N_BLOCKS / 1e3,
        "raw_samples_ms": raw,
    }


def report_row(row: dict) -> None:
    print(
        f"--- {row['res']}  (S={row['seq']})  "
        f"linear {row['linear_tflop_per_block']:.2f} TFLOP, SDPA {row['sdpa_tflop_per_block']:.3f} TFLOP ---"
    )
    print(f"  whole block            {row['block_ms']:8.2f} ms   100.0%")
    print(
        f"  linear only            {row['linear_ms']:8.2f} ms   {row['linear_share_pct']:5.1f}%"
        f"   ({row['linear_tflops']:.1f} TFLOP/s)"
    )
    print(
        f"  SDPA masked            {row['sdpa_masked_ms']:8.2f} ms   {row['sdpa_share_pct']:5.1f}%"
        f"   ({row['sdpa_tflops']:.1f} TFLOP/s)"
    )
    print(
        f"  SDPA unmasked          {row['sdpa_unmasked_ms']:8.2f} ms   {100 * row['sdpa_unmasked_ms'] / row['block_ms']:5.1f}%"
    )
    print(
        f"  everything else        {row['other_ms']:8.2f} ms   {row['other_share_pct']:5.1f}%   (block - linear - SDPA)"
    )
    print(
        f"  block with mask=None   {row['block_nomask_ms']:8.2f} ms"
        f"   -> mask costs {row['mask_cost_in_block_pct']:+.2f}% in-block"
    )
    print(f"  x {N_BLOCKS} blocks      = {row['step_s_34_blocks']:.2f} s per DiT step\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--bits", default="16,8", help="comma list: 16 = bf16, 8 = q8 as -q 8 loads it")
    parser.add_argument(
        "--grids", default="", help="substring filter over GRIDS keys; graph compile dominates the cost"
    )
    parser.add_argument("--json", default="block_linear_share.json")
    args = parser.parse_args()

    grids = {k: v for k, v in GRIDS.items() if not args.grids or args.grids in k}
    if not grids:
        raise SystemExit(f"--grids {args.grids!r} matched none of {list(GRIDS)}")

    stamp = env_stamp()
    print(f"mlx {stamp['mlx']}  python {stamp['python']}  commit {stamp['commit_short']} ({stamp['branch']})")
    print(f"activation precision {DT}  dim {DIM}  heads {HEADS}  hidden {HIDDEN}")
    print(f"random weights; {args.reps} timed reps per variant, median; grids {list(grids)}\n")

    out = {
        **stamp,
        "reps": args.reps,
        "order_policy": "variant order alternates forward and reverse on each repetition; raw samples are in raw_samples_ms",
        "runs": [],
    }
    for bits in [int(b) for b in args.bits.split(",")]:
        block = ZImageTransformerBlock(dim=DIM, n_heads=HEADS, norm_eps=1e-5, qk_norm=True)
        # The loader casts every float parameter to ModelConfig.precision (weight_loader.py:486), then
        # WeightApplier calls nn.quantize(model, class_predicate=..., bits=...) (weight_applier.py:123).
        block.set_dtype(DT)
        if bits != 16:
            nn.quantize(block, bits=bits)
        mx.eval(block.parameters())
        nbytes = param_bytes(block)
        tag = "bf16" if bits == 16 else f"q{bits}"
        print(
            f"##########  weights {tag}: {nbytes / 1e6:.0f} MB per block, "
            f"{nbytes * N_BLOCKS / 1e9:.2f} GB read per step  ##########\n"
        )
        rows = []
        for label, (h_tok, w_tok) in grids.items():
            row = measure(block, label, h_tok, w_tok, args.reps)
            rows.append(row)
            report_row(row)
        out["runs"].append({"weights": tag, "weight_mb_per_block": nbytes / 1e6, "rows": rows})

    with open(args.json, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
