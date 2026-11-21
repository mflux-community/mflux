"""Z-Image Turbo generation with the row-split ANE MLP behind a flag.

Reuses the real CLI's parser, model loader and latent creator, so bf16 / q8 / ANE differ only by
the patch this script applies. Nothing in src/mflux is touched.

  .venv/bin/python spikes/ane_matmul/stage3d/generate_ane.py \
      --prompt "..." --width 1024 --height 1024 --steps 8 --seed 0 \
      --ane-mlp --ane-share 0.50 --output stage3d/images/0_ane.png

Omit --ane-mlp for a baseline arm through the identical code path.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SPIKE = Path(__file__).resolve().parent
if str(SPIKE) not in sys.path:
    sys.path.insert(0, str(SPIKE))

from mflux.models.z_image.cli.z_image_turbo_generate import ZImageTurboCommand, build_parser  # noqa: E402
from mflux.utils.prompt_util import PromptUtil  # noqa: E402


def _read(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout  # noqa: S603
    except Exception:  # noqa: BLE001 -- a missing sysctl just means no memory column
        return ""


def swap_used_mb() -> float:
    m = re.search(r"used = ([\d.]+)M", _read(["sysctl", "-n", "vm.swapusage"]))
    return float(m.group(1)) if m else float("nan")


def wired_gb() -> float:
    out = _read(["vm_stat"])
    page = re.search(r"page size of (\d+)", out)
    wired = re.search(r"Pages wired down:\s+(\d+)", out)
    if not page or not wired:
        return float("nan")
    return int(wired.group(1)) * int(page.group(1)) / 2**30


def add_ane_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("stage 3d ANE MLP spike")
    g.add_argument(
        "--ane-mlp", action="store_true", help="route the main blocks' MLP through the row-split ANE program"
    )
    g.add_argument(
        "--ane-share", type=float, default=0.50, help="fraction of token rows given to the ANE (best from T1-T4: 0.50)"
    )
    g.add_argument("--ane-seq", type=int, default=4224, help="token count the split is sized for (1024x1024 -> 4224)")
    g.add_argument("--ane-backend", choices=["bridge", "mirror"], default="bridge")
    g.add_argument("--ane-surfaces", choices=["per-block", "shared"], default="per-block")
    g.add_argument(
        "--ane-no-range-fix",
        action="store_true",
        help="leave out the 1/sqrt(K) + h-normalisation fix; unsafe on trained weights, kept for comparison",
    )
    g.add_argument("--arm-name", default=None, help="label recorded in the JSON summary")


def main() -> None:
    parser = build_parser()
    add_ane_arguments(parser)
    args = parser.parse_args()

    wired0, swap0 = wired_gb(), swap_used_mb()
    t_load0 = time.perf_counter()
    model = ZImageTurboCommand.load(args)
    # a duration, not an absolute clock: perf_counter() has no fixed origin, so recording it raw
    # produced load_s values like 2186 that looked like a 36-minute load and were meaningless
    load_s = time.perf_counter() - t_load0
    # Sample at the phase boundary. Paging during load/encode is not the same validity problem as
    # paging inside the timed denoising window: under --low-ram the encoder is evicted right after
    # encoding, so whatever it pushed to swap may never be touched again. Splitting the delta lets
    # the gate look at the window that actually produced the timings.
    swap_after_load = swap_used_mb()
    wired_after_load = wired_gb()

    info: dict = {"arm": args.arm_name or ("ane" if args.ane_mlp else "baseline"), "quantize": args.quantize}
    if args.ane_mlp:
        import ane_patch

        info["patch"] = ane_patch.patch_model(
            model,
            share=args.ane_share,
            seq_len=args.ane_seq,
            backend_name=args.ane_backend,
            range_fix=not args.ane_no_range_fix,
            shared_surfaces=args.ane_surfaces == "shared",
        )

    seed = args.seed[0] if isinstance(args.seed, list) else args.seed
    prompt = PromptUtil.read_prompt(args)
    t0 = time.perf_counter()
    image = ZImageTurboCommand.generate(model, args, seed, prompt)
    gen_s = time.perf_counter() - t0
    swap_after_gen = swap_used_mb()
    steps = args.steps

    # the real CLI names this --output and formats it with the seed; reuse it unchanged
    out = Path(args.output.format(seed=seed) if "{seed}" in args.output else args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(path=str(out), export_json_metadata=args.metadata)

    # Record what the split actually saw before info goes to JSON: the real token count sets the
    # real share, and an image time quoted against the intended share would be misleading.
    runtime = info.get("patch", {}).pop("_runtime", None)
    observed: dict = {}
    if runtime is not None:
        st, be, sn = runtime["stats"], runtime["backend"], runtime["spec_name"]
        seqs = st.seq_seen()
        models = getattr(be, "_models", {})
        observed = {
            "seq_seen": seqs,
            "actual_share": [round(info["patch"]["mpad"] / s, 4) for s in seqs],
            "per_call_ms": {k: round(v, 3) for k, v in st.summary_ms().items()},
            "copybacks": models[sn].copybacks() if sn in models else None,
            "plan_report": getattr(be, "plan_reports", {}).get(sn),
            "ane_calls": len(st.wait),
        }

    summary = {
        **info,
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "width": args.width,
        "height": args.height,
        "ane_observed": observed or None,
        "load_s": round(load_s, 2),
        "image_s": round(gen_s, 3),
        "per_step_s": round(gen_s / steps, 4) if steps else None,
        "swap_mb_before": swap0,
        "swap_mb_after_load": swap_after_load,
        "swap_mb_after_gen": swap_after_gen,
        "swap_delta_load_mb": round(swap_after_load - swap0, 1),
        "swap_delta_window_mb": round(swap_after_gen - swap_after_load, 1),
        "wired_gb_before": round(wired0, 2),
        "wired_gb_after_load": round(wired_after_load, 2),
        "wired_gb_after": round(wired_gb(), 2),
        "path": str(out),
    }
    print("ANE-RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
