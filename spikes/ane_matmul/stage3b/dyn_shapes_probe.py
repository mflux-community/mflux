"""3b-2 dynamic-weights, phase 2: the four production call sites, dynamic vs baked.

Both sides are built HERE from the SAME synthetic int8 weights so the control
is exact. Dynamic form: mb.conv with the weight tensor as an fp16 INPUT (int8
codes carried exactly) plus tx/sw inputs; no K-split needed. Control: the
baked constexpr_blockwise conv at the identical fixed tier (4608).

Gate input: total ANE ms per block dynamic vs baked, macmon ANE peak proving
both run on the ANE, and copy vs wrap handoff cost.
"""

import json
import shutil
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dyn_ops_probe2 as D
import wired_probe_models as WP

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/ane3a/cache_dyn3")
RESULTS = HERE / "results"
T = 4608
SITES = [("qkv", 5760, 3840), ("o", 1920, 3840), ("w13", 10240, 3840), ("w2", 1920, 10240)]


def emit(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "dyn_shapes.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def build_dyn(O, I, int8_in=False):
    wdt = types.int8 if int8_in else types.fp16

    @mb.program(
        input_specs=[
            mb.TensorSpec((T, I), dtype=types.fp16),
            mb.TensorSpec((O, I), dtype=wdt),
            mb.TensorSpec((T,), dtype=types.fp16),
            mb.TensorSpec((O,), dtype=types.fp16),
        ],
        opset_version=D.TARGET,
    )
    def prog(x, w, tx, sw):
        wd = mb.cast(x=w, dtype="fp16") if int8_in else w
        xcf = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, I, 1, -1])
        y = mb.conv(x=xcf, weight=mb.reshape(x=wd, shape=[O, I, 1, 1]), pad_type="valid")
        y2 = mb.transpose(x=mb.reshape(x=y, shape=[O, -1]), perm=[1, 0])
        return mb.mul(x=mb.mul(x=y2, y=mb.expand_dims(x=tx, axes=[1])), y=mb.reshape(x=sw, shape=[1, O]), name="y")

    return prog, [ct.TensorType(name=n) for n in ("x", "w", "tx", "sw")]


def synth(O, I, seed):
    rng = np.random.default_rng(seed)
    w_true = (rng.standard_normal((O, I)) * 0.02).astype(np.float32)
    sw = np.maximum(np.abs(w_true).max(axis=1) / 127.0, np.float32(1e-8))
    wq = np.clip(np.round(w_true / sw[:, None]), -127, 127).astype(np.int8)
    x_true = (rng.standard_normal((T, I)) * 0.05).astype(np.float32)
    tx = np.maximum(np.abs(x_true).max(axis=1) / 127.0, np.float32(1e-8))
    xq = np.clip(np.round(x_true / tx[:, None]), -127, 127).astype(np.int8)
    ref = (
        (xq.astype(np.int64) @ wq.astype(np.int64).T).astype(np.float64)
        * tx.astype(np.float64)[:, None]
        * sw.astype(np.float64)[None, :]
    )
    return wq, sw, xq, tx, ref


def convert(prog, ci, path):
    if path.exists():
        shutil.rmtree(path)
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=ci,
        minimum_deployment_target=D.TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    m.save(str(path))


def cos_of(y, ref):
    yy = y.astype(np.float64).reshape(ref.shape)
    return float(yy.ravel() @ ref.ravel() / (np.linalg.norm(yy.ravel()) * np.linalg.norm(ref.ravel()) + 1e-30))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    for seed, (site, O, I) in enumerate(SITES, start=101):
        rec = {"site": site, "shape": [T, O, I]}
        wq, sw, xq, tx, ref = synth(O, I, seed)
        xdq = (xq.astype(np.float32) * tx[:, None]).astype(np.float16)
        pb = CACHE / f"baked_{site}.mlpackage"
        prog, ci = WP.build_fixed(wq, sw.astype(np.float16), "w8a16", ksplit=1, rows=T)
        convert(prog, ci, pb)
        res, err, y, peak = D.run_mode(pb, {"x": xdq}, [])
        if res:
            rec.update(baked_ms=round(res["med_ms"], 2), baked_cos=round(cos_of(y, ref), 7), baked_w=round(peak, 1))
        else:
            rec.update(baked="fail", baked_err=(err or "")[:160])
        pd = CACHE / f"dynf16_{site}.mlpackage"
        convert(*build_dyn(O, I), path=pd)
        dargs = {
            "x": xq.astype(np.float16),
            "w": wq.astype(np.float16),
            "tx": tx.astype(np.float16),
            "sw": sw.astype(np.float16),
        }
        for label, mode in (("dyn_copy", []), ("dyn_wrap", ["--wrap"])):
            res, err, y, peak = D.run_mode(pd, dargs, mode)
            if res is None:
                rec[label] = "fail"
                rec[label + "_err"] = (err or "")[:160]
                continue
            rec[label + "_ms"] = round(res["med_ms"], 2)
            rec[label + "_cos"] = round(cos_of(y, ref), 7)
            rec[label + "_w"] = round(peak, 1)
        if site == "qkv":  # the int8-input form, measured once for the table
            pi = CACHE / "dyni8_qkv.mlpackage"
            convert(*build_dyn(O, I, int8_in=True), path=pi)
            res, err, y, peak = D.run_mode(pi, {**dargs, "w": wq}, [])
            rec["dyni8_ms"] = round(res["med_ms"], 2) if res else "fail"
            rec["dyni8_cos"] = round(cos_of(y, ref), 7) if res else None
            rec["dyni8_w"] = round(peak, 1) if res else None
        emit(rec)


if __name__ == "__main__":
    main()
