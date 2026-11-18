"""Layer-level correctness + timing of the split w2 (block 15) on captured real activations.
Usage: MFLUX_ANE_LINEAR=1 check_layer.py ACTDIR [prompts]"""

import json
import sys
import time

import ane_split as A
import mlx.core as mx
import numpy as np

ACT = sys.argv[1]
N_PROMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
W = A.load_bf16_weight("layers.15.feed_forward.w2.weight")
ref_w = mx.array(W).astype(mx.bfloat16)
res = {}
for share in (1.0, 0.0, 0.5, 0.6):
    lin = A.AneSplitLinear(W, share, A.HERE / "cache", "b15w2")
    for p in range(N_PROMPTS):
        x = mx.array(np.load(f"{ACT}/p{p}_w2.npy")).astype(mx.bfloat16)[None]
        y = lin(x)
        mx.eval(y)  # warm-up
        for _ in range(5):
            mx.eval(lin(x))
        ts = []
        for _ in range(30):
            t = time.perf_counter()
            mx.eval(lin(x))
            ts.append(time.perf_counter() - t)
        ref = np.array((x.reshape(-1, W.shape[1]).astype(mx.float32)) @ mx.array(W).T)
        out = np.array(y.astype(mx.float32)).reshape(ref.shape)
        cos = float((out * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref)))
        r = dict(
            share=share, prompt=p, median_ms=float(np.median(ts) * 1e3), cos=cos, max_abs=float(np.abs(out - ref).max())
        )
        print(json.dumps(r), flush=True)
