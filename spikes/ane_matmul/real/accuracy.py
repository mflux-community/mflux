"""Accuracy of q8 / W8A16 / W8A8 vs fp32 reference on captured real activations (block 15).
Usage: accuracy.py ACTDIR"""

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import coremltools as ct
import gen_models as G
import mlx.core as mx
import numpy as np
from safetensors import safe_open

ACT = sys.argv[1]
from ckpt import find_transformer_dir

snap = find_transformer_dir()
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
LAYERS = {
    "to_q": "layers.15.attention.to_q.weight",
    "w1": "layers.15.feed_forward.w1.weight",
    "w2": "layers.15.feed_forward.w2.weight",
}
idx = json.load(open(snap + "diffusion_pytorch_model.safetensors.index.json"))["weight_map"]


def metrics(y, ref):
    y = y.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    cos = float(y @ r / (np.linalg.norm(y) * np.linalg.norm(r) + 1e-30))
    return cos, float(np.abs(y - r).max()), float(np.abs(y - r).max() / (np.abs(r).max() + 1e-30))


import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="aneacc_")
import atexit
import shutil

atexit.register(shutil.rmtree, TMP, ignore_errors=True)


def ane(prog, xin, O, T):
    """Convert, save, run once through the Swift runner (Python predict cannot take int8 inputs)."""
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=G.TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    mp = f"{TMP}/m.mlpackage"
    m.save(mp)
    paths = []
    for i, arr in enumerate(xin):
        pth = f"{TMP}/in{i}.bin"
        np.ascontiguousarray(arr).tofile(pth)
        paths.append(pth)
    subprocess.run([HERE + "/../bench/anerun", mp, f"{TMP}/out.bin", *paths], check=True, timeout=300)
    return np.fromfile(f"{TMP}/out.bin", dtype=np.float16).reshape(O, T)


rows = []
for name, key in LAYERS.items():
    if ONLY and name not in ONLY:
        continue
    PRE = 1 / 16 if name == "w2" else 1.0  # fp16 headroom: w2 outputs reach ~80k > 65504
    with safe_open(snap + idx[key], "pt") as f:
        W = f.get_tensor(key).float().numpy()  # bf16 original, as fp32
    O, I = W.shape
    sc = (np.abs(W).max(1) / 127.0).astype(np.float16)  # symmetric per-output-channel
    Wq = np.clip(np.round(W / sc.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
    Wb = mx.array(W).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(Wb, group_size=64, bits=8)
    ks = 4 if name == "w2" else 1
    for p in range(3):
        x = np.load(f"{ACT}/p{p}_{name}.npy").astype(np.float32)
        T = x.shape[0]
        ref = x @ W.T
        xb = mx.array(x).astype(mx.bfloat16)
        out = {
            "bf16": np.array((xb @ Wb.T).astype(mx.float32)),
            "q8": np.array(
                mx.quantized_matmul(xb, qw, qs, qb, transpose=True, group_size=64, bits=8).astype(mx.float32)
            ),
        }
        xc = (x * PRE).T.reshape(1, I, 1, T).astype(np.float16)
        y = ane(G.build_conv(T, O, I, ksplit=ks, w_s=(Wq, sc)), [xc], O, T)
        out["W8A16"] = y.T.astype(np.float32) / PRE
        # W8A8: per-token symmetric int8 activations, scale applied after the matmul
        tokf = np.maximum(np.abs(x).max(1) / 127.0, 1e-8)
        tok = (tokf * PRE).astype(np.float16)
        xi = np.clip(np.round(x / tokf.astype(np.float16).astype(np.float32)[:, None]), -127, 127).astype(np.int8)
        y = ane(
            G.build_conv(T, O, I, ksplit=ks, a8="pre", w_s=(Wq, sc)),
            [xi.T.reshape(1, I, 1, T), tok.reshape(1, 1, 1, T)],
            O,
            T,
        )
        out["W8A8"] = y.T.astype(np.float32) / PRE
        for k, v in out.items():
            cos, mad, rel = metrics(v, ref)
            rows.append(
                dict(
                    layer=name,
                    prompt=p,
                    variant=k,
                    cos=cos,
                    max_abs=mad,
                    rel_max=rel,
                    ref_absmax=float(np.abs(ref).max()),
                )
            )
            print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(f"{ACT}/accuracy_{'_'.join(ONLY or ['all'])}.json", "w"))
