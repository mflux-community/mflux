"""Test 2: GPU+ANE concurrency splits for every production shape.

Shapes: o (3840x3840), qkv fused (11520x3840), w1w3 fused (20480x3840), w2 (3840x10240, K-split 4).
GPU:ANE column shares 60:40, 50:50, 45:55, 40:60, plus 100:0 and 0:100 references.
The GPU half runs in q8 and in bf16. Method: both halves loop in a shared time
window (Stage 1 method). layer_ms = max(gpu_ms, ane_ms) when the halves overlap.
"""

import json
import os
import subprocess
import time

import s3a
from s3a import ANEBENCH, MODELS, PYTHON, WATCHDOG, BenchLock, emit, gen_fixed, run

T = s3a.T_FIXED
RAMP_S, WINDOW_S = 12, 8
SHAPES = [
    ("o", 3840, 3840, "c1"),
    ("qkv", 11520, 3840, "c1"),
    ("w1w3", 20480, 3840, "c1"),
    ("w2", 3840, 10240, "c4"),
]
GPU_MODES = ["q8", "bf16"]
ANE_SHARES = [0.4, 0.5, 0.55, 0.6]  # GPU shares are 0.6, 0.5, 0.45, 0.4


def round64(v):
    return max(64, int(round(v / 64)) * 64)


def gpu_solo(O, I, mode):
    out = run([PYTHON, os.path.join(os.path.dirname(__file__), "gpu_half.py"), T, O, I, mode, 0, 0])
    return json.loads(out.strip().splitlines()[-1])["median_ms"]


def concurrent(O_gpu, O_ane, I, mode, variant):
    """Run the GPU half and the ANE half in one shared window; return both medians."""
    start = time.time() + RAMP_S
    end = start + WINDOW_S
    ane_model = MODELS / f"{variant}_{T}x{O_ane}x{I}.mlpackage"
    ane_cmd = [
        WATCHDOG,
        ANEBENCH,
        str(ane_model),
        "--units",
        "ane",
        "--shape",
        str(T),
        str(O_ane),
        str(I),
        "--layout",
        "conv",
        "--window",
        str(start),
        str(end),
    ]
    gpu_cmd = [
        WATCHDOG,
        PYTHON,
        os.path.join(os.path.dirname(__file__), "gpu_half.py"),
        T,
        O_gpu,
        I,
        mode,
        str(start),
        str(end),
    ]
    pa = subprocess.Popen([str(c) for c in ane_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pg = subprocess.Popen([str(c) for c in gpu_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    og, eg = pg.communicate(timeout=300)
    oa, ea = pa.communicate(timeout=300)
    if pg.returncode != 0 or pa.returncode != 0:
        raise RuntimeError(f"concurrent run failed: gpu({pg.returncode}) {eg[-500:]} ane({pa.returncode}) {ea[-500:]}")
    gpu_ms = json.loads(og.strip().splitlines()[-1])["median_ms"]
    ane_ms = json.loads(oa.strip().splitlines()[-1])["median_ms"]
    return gpu_ms, ane_ms


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    # Pre-generate every ANE model the sweep needs (generation is not timed).
    for _, O, I, variant in SHAPES:
        gen_fixed(MODELS, T, O, I, variant)  # 0:100 reference
        for share in ANE_SHARES:
            gen_fixed(MODELS, T, round64(O * share), I, variant)
    with BenchLock(owner=f"stage3a-t2-{os.getpid()}"):
        for name, O, I, variant in SHAPES:
            for mode in GPU_MODES:
                solo = gpu_solo(O, I, mode)
                emit(
                    "t2_splits",
                    dict(
                        shape=name,
                        gpu_mode=mode,
                        gpu_share=1.0,
                        ane_share=0.0,
                        gpu_ms=solo,
                        ane_ms=None,
                        layer_ms=solo,
                        speedup=1.0,
                    ),
                )
                ane_full = s3a.anebench(MODELS / f"{variant}_{T}x{O}x{I}.mlpackage", shape=(T, O, I), iters=50)
                emit(
                    "t2_splits",
                    dict(
                        shape=name,
                        gpu_mode=mode,
                        gpu_share=0.0,
                        ane_share=1.0,
                        gpu_ms=None,
                        ane_ms=ane_full["median_ms"],
                        layer_ms=ane_full["median_ms"],
                        speedup=solo / ane_full["median_ms"],
                    ),
                )
                for ane_share in ANE_SHARES:
                    O_ane = round64(O * ane_share)
                    O_gpu = O - O_ane
                    gpu_ms, ane_ms = concurrent(O_gpu, O_ane, I, mode, variant)
                    layer = max(gpu_ms, ane_ms)
                    emit(
                        "t2_splits",
                        dict(
                            shape=name,
                            gpu_mode=mode,
                            gpu_share=O_gpu / O,
                            ane_share=O_ane / O,
                            o_gpu=O_gpu,
                            o_ane=O_ane,
                            gpu_ms=gpu_ms,
                            ane_ms=ane_ms,
                            layer_ms=layer,
                            speedup=solo / layer,
                        ),
                    )


if __name__ == "__main__":
    main()
