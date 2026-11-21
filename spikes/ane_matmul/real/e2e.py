"""End-to-end: Z-Image Turbo q8, 1024x1024, 8 steps, fixed seeds. Flag off vs on (block 15 w2 split 50:50).
Usage: e2e.py OUTDIR [gpu_share]"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["MFLUX_ANE_LINEAR"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ane_linear"))
import ane_split

from mflux.models.z_image.variants.z_image import ZImage

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SHARE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
MODEL = "/Volumes/Extreme2Tb/MFlux-Models/Z-Image-Turbo-mflux-q8"
PROMPTS = [
    "A red fox sitting in a snowy forest at dawn, photorealistic",
    "A futuristic city skyline at night with neon signs, cinematic",
    "A bowl of ripe fruit on a wooden table, oil painting style",
]
model = ZImage(model_path=MODEL)


def gen(i, tag):
    t = time.time()
    img = model.generate_image(seed=100 + i, prompt=PROMPTS[i], num_inference_steps=8, height=1024, width=1024)
    dt = time.time() - t
    img.image.save(OUT / f"{tag}_p{i}.png")
    return np.array(img.image).astype(np.float64), dt


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return None if mse == 0 else float(10 * np.log10(255.0**2 / mse))  # None = identical images (JSON-safe)


gen(0, "warmup_off")  # warm-up (not recorded)
off = [gen(i, "off") for i in range(3)]
ane_split.apply_ane_split(model, block=15, gpu_share=SHARE)
assert isinstance(model.transformer.layers[15].feed_forward.w2, ane_split.AneSplitLinear)
gen(0, "warmup_on")  # warm-up with the patch
on = [gen(i, "on") for i in range(3)]
rows = [
    dict(
        prompt=i,
        off_s=off[i][1],
        on_s=on[i][1],
        off_step_s=off[i][1] / 8,
        on_step_s=on[i][1] / 8,
        psnr_db=psnr(off[i][0], on[i][0]),
    )
    for i in range(3)
]
print(json.dumps(rows, indent=1))
json.dump(rows, open(OUT / "e2e.json", "w"), indent=1)
