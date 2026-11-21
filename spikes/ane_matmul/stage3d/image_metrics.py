#!/usr/bin/env python
"""Image arm comparison metrics for Stage 3d Step 3.

  .venv/bin/python spikes/ane_matmul/stage3d/image_metrics.py --ref images/1_bf16.png --arms images/1_q8.png images/1_ane.png

PSNR is computed here in numpy. LPIPS is only attempted if the `lpips` package is importable --
it is NOT installed in this venv, and installing it pulls torchvision VGG weights, so that needs
Scriv's approval first. Without it the table records psnr and leaves lpips as null rather than
substituting a different metric silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}: different image sizes are not comparable")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def try_lpips(a: np.ndarray, b: np.ndarray):
    try:
        import lpips  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- absence is the expected state here
        return None
    global _LPIPS_MODEL
    if "_LPIPS_MODEL" not in globals():
        _LPIPS_MODEL = lpips.LPIPS(net="vgg")
    t = torch.tensor(np.stack([a, b])).permute(0, 3, 1, 2) * 2.0 - 1.0
    with torch.no_grad():
        return float(_LPIPS_MODEL(t[0:1], t[1:2]).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="baseline image (bf16)")
    ap.add_argument("--arms", nargs="+", required=True, help="candidate images to score against --ref")
    ap.add_argument("--out", default=None, help="optional JSON path for the rows")
    args = ap.parse_args()

    ref = load_rgb(args.ref)
    rows = []
    print(f"ref={args.ref}  shape={ref.shape}")
    for arm in args.arms:
        img = load_rgb(arm)
        p = psnr(ref, img)
        lp = try_lpips(ref, img)
        exact = bool(np.array_equal(ref, img))
        rows.append({"arm": arm, "psnr_vs_ref": p, "lpips_vs_ref": lp, "identical": exact})
        print(f"  {arm:>34}  psnr={p:>8.2f}  lpips={'n/a' if lp is None else f'{lp:.5f}'}  identical={exact}")
    if args.out:
        Path(args.out).write_text(json.dumps({"ref": args.ref, "rows": rows}, indent=2))
        print("wrote", args.out)
    if any(r["lpips_vs_ref"] is None for r in rows):
        print("\nLPIPS not computed: the lpips package is absent from .venv. Installing it pulls VGG")
        print("weights over the network, which needs Scriv's approval -- do not install unasked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
