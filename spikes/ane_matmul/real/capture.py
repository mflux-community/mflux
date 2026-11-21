"""Capture real Z-Image Turbo (q8) activations at DiT block 15 for 3 prompts.
Saves fp16 .npy of the inputs to to_q, w1 and w2 (denoise step 4 of 8, 1024x1024), plus abs-max over all steps.
Usage: capture.py OUTDIR"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.compile = lambda f=None, *a, **k: f if f is not None else (lambda g: g)  # hooks need eager arrays
import mlx.nn as nn

from mflux.models.z_image.variants.z_image import ZImage

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "/Volumes/Extreme2Tb/MFlux-Models/Z-Image-Turbo-mflux-q8"
PROMPTS = [
    "A red fox sitting in a snowy forest at dawn, photorealistic",
    "A futuristic city skyline at night with neon signs, cinematic",
    "A bowl of ripe fruit on a wooden table, oil painting style",
]


class Rec(nn.Module):
    def __init__(self, inner, name, store):
        super().__init__()
        self.inner, self.name, self.store = inner, name, store

    def __call__(self, x):
        mx.eval(x)
        self.store.setdefault(self.name, []).append(x)
        return self.inner(x)


model = ZImage(model_path=MODEL)
blk = model.transformer.layers[15]
store = {}
blk.attention.to_q = Rec(blk.attention.to_q, "to_q", store)
blk.feed_forward.w1 = Rec(blk.feed_forward.w1, "w1", store)
blk.feed_forward.w2 = Rec(blk.feed_forward.w2, "w2", store)


def run(prompt, seed, steps=8, size=1024):
    store.clear()
    t = time.time()
    model.generate_image(seed=seed, prompt=prompt, num_inference_steps=steps, height=size, width=size)
    return time.time() - t


print("warm-up", run("warm up", 0, steps=2, size=512), flush=True)
stats = {}
for i, p in enumerate(PROMPTS):
    dt = run(p, 100 + i)
    for name, xs in store.items():
        arrs = [np.array(x.astype(mx.float32)) for x in xs]
        step = arrs[min(3, len(arrs) - 1)]
        if np.abs(step).max() > 65504:
            raise ValueError(f"{name}: activation {np.abs(step).max():.0f} does not fit fp16; scale before saving")
        np.save(OUT / f"p{i}_{name}.npy", step.reshape(-1, step.shape[-1]).astype(np.float16))
        stats[f"p{i}_{name}"] = dict(
            calls=len(arrs), shape=list(step.shape), absmax_all_steps=float(max(np.abs(a).max() for a in arrs))
        )
    print("prompt", i, f"{dt:.1f}s", flush=True)
json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
