#!/usr/bin/env python
"""Side-by-side labelled grids for the Stage 3d eyeball check.

  .venv/bin/python spikes/ane_matmul/stage3d/make_grids.py --dir spikes/ane_matmul/stage3d/images --indices 1 2 3 4

For each prompt index i it lays the arm images out left to right as `<i>_grid.png`, with the arm
name drawn under each panel and the prompt drawn along the top. Panels are scaled to a common
height so mixed sizes still compose; the original files are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABEL_H, HEADER_H, GAP, BORDER = 40, 64, 12, 2


def _font(size: int):
    # DejaVu ships with Pillow on macOS wheels; fall back to the default bitmap font if absent
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        p = Path(name)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:  # unreadable font file, keep trying
                continue
    return ImageFont.load_default()


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_grid(paths: list[Path], labels: list[str], prompt: str, out: Path, target_h: int = 512) -> dict:
    if len(paths) != len(labels):
        raise ValueError(f"{len(paths)} images but {len(labels)} labels")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    scaled = []
    for im in imgs:
        w = max(1, round(im.width * target_h / im.height))
        scaled.append(im.resize((w, target_h), Image.LANCZOS))
    total_w = sum(im.width for im in scaled) + GAP * (len(scaled) + 1) + BORDER * 2 * len(scaled)
    canvas = Image.new("RGB", (total_w, target_h + LABEL_H + HEADER_H), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    big, small = _font(22), _font(18)
    draw.text((GAP, 10), _shorten(f"{prompt}", 150), font=big, fill=(240, 240, 240))
    x = GAP
    for label, im in zip(labels, scaled, strict=True):
        draw.rectangle(
            [x - BORDER, HEADER_H - BORDER, x + im.width + BORDER, HEADER_H + im.height + BORDER],
            outline=(90, 90, 90),
            width=BORDER,
        )
        canvas.paste(im, (x, HEADER_H))
        draw.text((x, HEADER_H + im.height + 8), label, font=small, fill=(220, 220, 220))
        x += im.width + GAP
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out))
    return {"grid": str(out), "panels": labels, "width": canvas.width, "height": canvas.height}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding <i>_<arm>.png")
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["bf16", "q8", "ane"])
    ap.add_argument("--prompts", default=None, help="JSON file with a list of prompts, index 1 first")
    ap.add_argument("--target-height", type=int, default=512)
    args = ap.parse_args()

    d = Path(args.dir)
    prompts = []
    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())
    rows, skipped = [], []
    for i in args.indices:
        paths, missing = [], []
        for arm in args.arms:
            p = d / f"{i}_{arm}.png"
            if p.exists():
                paths.append(p)
            else:
                missing.append(str(p))
        if missing:
            # A grid with a panel silently dropped would still look like a complete comparison, so
            # refuse it: report the gaps and fail at the end rather than emit a partial grid.
            skipped.append((i, missing))
            print(f"  prompt {i}: MISSING {', '.join(missing)} -- no grid written")
            continue
        prompt = prompts[i - 1] if len(prompts) >= i else f"prompt {i}"
        info = build_grid(paths, args.arms, prompt, d / f"{i}_grid.png", args.target_height)
        info["index"] = i
        rows.append(info)
        print(f"  wrote {info['grid']} ({info['width']}x{info['height']}, {len(args.arms)} panels)")
    (d / "grids.json").write_text(json.dumps({"grids": rows, "skipped": skipped}, indent=2))
    print(f"wrote {d / 'grids.json'}: {len(rows)} grids, {len(skipped)} skipped")
    return 0 if rows and not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
