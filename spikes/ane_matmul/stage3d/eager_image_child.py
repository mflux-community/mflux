"""One process = one arm. A = stock mflux (whole-step mx.compile). C = eager: per-block compile, threaded ANE MLP.

  python eager_image_child.py ARM REP OUTDIR     (ARM in A, C)
Prints one RESULT line with the timed seconds per prompt. Images are saved on REP 0.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
from eager_bench import SplitBlock, make_epilogue, real_ane_fns  # noqa: E402,F401
from mlp_program import MlpSpec, plan_split  # noqa: E402

from mflux.models.z_image.cli.z_image_turbo_generate import ZImageTurboCommand, build_parser  # noqa: E402
from mflux.models.z_image.variants.z_image import ZImage  # noqa: E402

arm, rep, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
# A trailing F selects the fused QK-norm + RoPE kernel (PR #1009); without it the composed ops are forced.
FUSED = arm.endswith("F") and arm != "C8K"
ARM_LABEL = arm
if FUSED:
    arm = arm[:-1]
else:
    os.environ["MFLUX_ZIMAGE_DISABLE_FUSED_QKNORM_ROPE"] = "1"
IS_Q8 = arm in ("A8", "C8", "C8K")  # GPU weights: q8 (local mflux q8 model) or bf16 (HF default)
IS_ANE = arm in ("C", "C8", "C8K")
DROP_ENCODER = arm != "C8K"
Q8_PATH = "/Volumes/Extreme2Tb/MFlux-Models/Z-Image-Turbo-mflux-q8"
PROMPTS = {
    "A": "a red fox sitting in a snowy forest at dawn, detailed fur, soft light",
    "B": "a neon-lit street market in the rain at night, people with umbrellas, cinematic",
}


def parse(steps):
    sys.argv = ["x", "--prompt", "x", "--steps", str(steps), "--width", "1024", "--height", "1024", "--seed", "1"]
    if IS_Q8:
        sys.argv += ["--model", Q8_PATH]
    return build_parser().parse_args()


class EagerLayer:
    """Plain-Python stand-in for a block inside the transformer: compiled per block, same call signature."""

    def __init__(self, block):
        self.c = mx.compile(lambda x, mask, freqs, t: block(x, mask, freqs, t))

    def __call__(self, x, attn_mask, freqs_cis, t_emb):
        return self.c(x, attn_mask, freqs_cis, t_emb)


class RefinerLayer:
    def __init__(self, block):
        self.block, self.has_t = block, "t_emb" in block.__call__.__code__.co_varnames
        if self.has_t:
            self.c = mx.compile(lambda x, mask, freqs, t: block(x, mask, freqs, t))
        else:
            self.c = mx.compile(lambda x, mask, freqs: block(x, mask, freqs))

    def __call__(self, x, attn_mask, freqs_cis, t_emb=None):
        return self.c(x, attn_mask, freqs_cis, t_emb) if self.has_t else self.c(x, attn_mask, freqs_cis)


class SplitLayer:
    def __init__(self, sb):
        self.sb = sb

    def __call__(self, x, attn_mask, freqs_cis, t_emb):
        return self.sb(x, attn_mask, freqs_cis, t_emb)


def ane_fns_from_safetensors(n, spec, cache):
    """q8 GPU weights are group-wise int8 and cannot feed the ANE, so rebuild the ANE's per-row int8 from the HF fp32 shards."""
    import numpy as np
    from rowsplit import BridgeBackend, prepare_block_weights

    repo = Path.home() / ".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo"
    snap = next(repo.glob("snapshots/*")).resolve() / "transformer"
    wmap = json.loads((snap / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    backend = BridgeBackend(cache)
    fns, gains = [], []
    for i in range(n):
        ws = []
        for nm in ("w1", "w3", "w2"):
            key = f"layers.{i}.feed_forward.{nm}.weight"
            t = mx.load(str(snap / wmap[key]))[key]
            ws.append(np.array(t.astype(mx.float32)))
        w = prepare_block_weights(spec, ws[0], ws[1], ws[2])
        fns.append(lambda x, w=w: backend.run(spec, w, x))
        gains.append(w.gain)
        del ws
    return fns, gains, backend


import re  # noqa: E402
import resource  # noqa: E402

from bench import MacmonSampler  # noqa: E402


def wired_gb():
    out = os.popen("vm_stat").read()
    page = re.search(r"page size of (\d+)", out)
    w = re.search(r"Pages wired down:\s+(\d+)", out)
    return round(int(w.group(1)) * int(page.group(1)) / 1e9, 2) if page and w else None


swap_start, wired_start = os.popen("sysctl -n vm.swapusage").read().strip(), wired_gb()
args = parse(8)
model = ZImageTurboCommand.load(args)
info = {"arm": ARM_LABEL}
# Memory: encode every prompt once, then drop the text encoder (about 8 GB) before the ANE weights are built.
# Both arms do this, so the timed window is denoise + VAE in each. Encoding is not in the timed region.
import gc  # noqa: E402

_enc = {}
for _p in [*PROMPTS.values(), "warmup"]:
    _enc[_p] = model._encode_prompts(prompt=_p, negative_prompt=None, guidance=0.0)
    mx.eval(_enc[_p][0])
if DROP_ENCODER:
    model.text_encoder = None
    gc.collect()
    mx.clear_cache()
model._encode_prompts = lambda *, prompt, negative_prompt, guidance: _enc[prompt]
info["text_encoder_dropped"] = DROP_ENCODER
if IS_ANE:
    seq_expected = 4224
    m, _ = plan_split(seq_expected, 0.50)
    tr = model.transformer
    phases = {"pre_wait": [], "gpu_enqueue": [], "join_wait": []}
    spec = MlpSpec(mpad=m, scale_mode="output", range_fix=True)
    if IS_Q8:
        fns, gains, backend = ane_fns_from_safetensors(len(tr.layers), spec, HERE / "cache")
    else:
        fns, gains, backend = real_ane_fns(list(tr.layers), spec, HERE / "cache")
    pool = ThreadPoolExecutor(max_workers=1)
    tr.noise_refiner = [RefinerLayer(b) for b in tr.noise_refiner]
    tr.context_refiner = [RefinerLayer(b) for b in tr.context_refiner]
    tr.layers = [SplitLayer(SplitBlock(b, fn, g, m, pool, phases)) for b, fn, g in zip(tr.layers, fns, gains)]

    def plain_predict(transformer):  # no whole-step compile: the threaded ANE call cannot live inside one
        def predict(latents, timestep, sigmas, text_encodings, negative_encodings, guidance):
            return transformer(timestep=timestep, x=latents, cap_feats=text_encodings, sigmas=sigmas)

        return predict

    ZImage._predict = staticmethod(plain_predict)
    info.update(mpad=m, plan_report=getattr(backend, "plan_reports", {}))

# warmup (compiles; 2 steps)
ZImageTurboCommand.generate(model, parse(2), 9, "warmup")
swap0 = os.popen("sysctl -n vm.swapusage").read().strip()
wired_before_gen = wired_gb()
mm = MacmonSampler()
mm.__enter__()
res = {}
for k, p in PROMPTS.items():
    if IS_ANE:
        for v in phases.values():
            v.clear()
    t = time.perf_counter()
    img = ZImageTurboCommand.generate(model, args, 1, p)
    mx.eval(mx.array(0))
    res[k] = time.perf_counter() - t
    if rep == "0":
        img.save(path=f"{outdir}/{ARM_LABEL}_{k}.png")
swap1 = os.popen("sysctl -n vm.swapusage").read().strip()
mm.__exit__()
if IS_ANE:
    import numpy as np

    info["phase_ms_median"] = {k: 1e3 * float(np.median(v)) for k, v in phases.items() if v}
    info["ane_calls_last_prompt"] = len(phases["join_wait"])
print(
    "RESULT "
    + json.dumps(
        {
            "arm": arm,
            "rep": rep,
            "times": res,
            "swap_before_gen": swap0,
            "swap_after": swap1,
            "swap_start": swap_start,
            "wired_gb": [wired_start, wired_before_gen, wired_gb()],
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
            "ane_w_median": mm.median_w(),
            **info,
        }
    )
)
