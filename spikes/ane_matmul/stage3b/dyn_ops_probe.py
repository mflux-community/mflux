"""3b-2 dynamic-weights ops probe: does Core ML 9 on macOS 26 run int8 weight-INPUT matmuls on the ANE?

Draw Things runs FLUX.2-klein on the ANE of this machine with int8 weights
passed as graph INPUTS (dynamic weights), matmul-only programs, ~22 TFLOPS.
Our Stage 1 V1 attempt was 2x slower than baked weights. This probe tests
which program forms this coremltools/OS combination accepts, whether the
numerics are int8 x int8 with int32-accurate accumulation, and how fast each
form runs, in three handoff modes: owned MLMultiArray per call, zero-copy
pointer wrap (--wrap), and CPU-only reference. Prediction runs through
bench/dynrun because coremltools' Python predict cannot feed int8 arrays.

Variants (x = activations, w = weights):
  mm_i8    int8 x, int8 w   -> mb.matmul, scales applied after on the cast output
  mm_f16   fp16 x, fp16 w   -> mb.matmul (ANE fp16 matmul reference with dynamic weights)
  conv_i8  int8 x, int8 w   -> mb.conv with a NON-CONSTANT weight input
  conv_dq  int8 x, int8 w   -> dequantise both in-graph then fp16 conv (old V1 form)
  baked    constexpr_blockwise weights (production control)

Checks per variant: convert OK, predict OK, cosine vs int64 emulation,
30-call median ms (ANE copy), (ANE wrap), CPU-only ms, macmon ane_power peak.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/ane3a/cache_dyn")
RESULTS = HERE / "results"
DYNRUN = HERE.parent / "bench" / "dynrun"

# small probe shapes
T, I, O = 320, 256, 192
rng = np.random.default_rng(1)
xq = rng.integers(-127, 128, (T, I)).astype(np.int8)
wq = rng.integers(-127, 128, (O, I)).astype(np.int8)
tx = (np.abs(xq.astype(np.float32)).max(axis=1) / 127.0 + 1e-6).astype(np.float32)
sw = (np.abs(wq.astype(np.float32)).max(axis=1) / 127.0 + 1e-6).astype(np.float32)
TRUE = (xq.astype(np.int64) @ wq.astype(np.int64).T).astype(np.float64)  # int32-accurate
TARGET = ct.target.iOS26 if hasattr(ct.target, "iOS26") else ct.target.iOS18


def emit(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "dyn_ops_probe.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def build_mm_i8():
    @mb.program(
        input_specs=[
            mb.TensorSpec((T, I), dtype=types.int8),
            mb.TensorSpec((O, I), dtype=types.int8),
            mb.TensorSpec((T,), dtype=types.fp16),
            mb.TensorSpec((O,), dtype=types.fp16),
        ],
        opset_version=TARGET,
    )
    def prog(x, w, tx_, sw_):
        acc = mb.matmul(x=x, y=mb.transpose(x=w, perm=[1, 0]))
        acc16 = mb.cast(x=acc, dtype="fp16")
        return mb.mul(x=mb.mul(x=acc16, y=mb.expand_dims(x=tx_, axes=[1])), y=mb.expand_dims(x=sw_, axes=[1]), name="y")

    return prog, [ct.TensorType(name=n) for n in ("x", "w", "tx", "sw")]


def build_mm_f16():
    @mb.program(
        input_specs=[mb.TensorSpec((T, I), dtype=types.fp16), mb.TensorSpec((O, I), dtype=types.fp16)],
        opset_version=TARGET,
    )
    def prog(x, w):
        return mb.matmul(x=x, y=mb.transpose(x=w, perm=[1, 0]), name="y")

    return prog, [ct.TensorType(name=n) for n in ("x", "w")]


def build_conv_i8():
    @mb.program(
        input_specs=[
            mb.TensorSpec((1, I, 1, T), dtype=types.int8),
            mb.TensorSpec((O, I, 1, 1), dtype=types.int8),
            mb.TensorSpec((1, 1, 1, T), dtype=types.fp16),
            mb.TensorSpec((1, O, 1, 1), dtype=types.fp16),
        ],
        opset_version=TARGET,
    )
    def prog(x, w, tx_, sw_):
        y = mb.conv(x=x, weight=w, pad_type="valid")
        y16 = mb.cast(x=y, dtype="fp16")
        return mb.mul(x=mb.mul(x=y16, y=tx_), y=sw_, name="y")

    return prog, [ct.TensorType(name=n) for n in ("x", "w", "tx", "sw")]


def build_conv_dq():
    @mb.program(
        input_specs=[
            mb.TensorSpec((1, I, 1, T), dtype=types.int8),
            mb.TensorSpec((O, I, 1, 1), dtype=types.int8),
            mb.TensorSpec((1, 1, 1, T), dtype=types.fp16),
            mb.TensorSpec((1, O, 1, 1), dtype=types.fp16),
        ],
        opset_version=TARGET,
    )
    def prog(x, w, tx_, sw_):
        xd = mb.dequantize(input=x, scale=mb.squeeze(x=tx_, axes=[0, 1, 2]))
        wd = mb.dequantize(input=w, scale=mb.squeeze(x=sw_, axes=[0, 2, 3]))
        return mb.conv(x=xd, weight=wd, pad_type="valid", name="y")

    return prog, [ct.TensorType(name=n) for n in ("x", "w", "tx", "sw")]


def build_baked():
    @mb.program(input_specs=[mb.TensorSpec((T, I), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        w4 = mb.constexpr_blockwise_shift_scale(
            data=np.ascontiguousarray(wq.reshape(O, I, 1, 1)), scale=sw.reshape(O, 1, 1, 1), offset=None
        )
        xd = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, I, 1, -1])
        return mb.identity(
            x=mb.transpose(x=mb.reshape(x=mb.conv(x=xd, weight=w4, pad_type="valid"), shape=[O, -1]), perm=[1, 0]),
            name="y",
        )

    return prog, [ct.TensorType(name="x", shape=[T, I])]


def inputs_for(name):
    """dict name -> (bytes, np array in the model's dtype)."""
    if name in ("mm_f16", "baked"):
        xdq = (xq.astype(np.float32) * tx[:, None]).astype(np.float16)
        out = {"x": xdq}
        if name == "mm_f16":
            out["w"] = (wq.astype(np.float32) * sw[:, None]).astype(np.float16)
        return out
    return {"x": xq, "w": wq, "tx": tx.astype(np.float16), "sw": sw.astype(np.float16)}


def reshape_for(name, arrs):
    if name == "conv_i8":
        arrs = dict(arrs)
        arrs["x"] = arrs["x"].reshape(1, I, 1, T)
        arrs["tx"] = arrs["tx"].reshape(1, 1, 1, T)
        arrs["sw"] = arrs["sw"].reshape(1, O, 1, 1)
    if name == "conv_dq":
        arrs = dict(arrs)
        arrs["x"] = arrs["x"].reshape(1, I, 1, T)
        arrs["tx"] = arrs["tx"].reshape(1, 1, 1, T)
        arrs["sw"] = arrs["sw"].reshape(1, O, 1, 1)
    return arrs


def ref_out(name):
    if name == "mm_f16":
        xdq = xq.astype(np.float32) * tx[:, None]
        wdq = wq.astype(np.float32) * sw[:, None]
        return xdq.astype(np.float64) @ wdq.astype(np.float64).T
    if name == "baked":
        xdq = xq.astype(np.float32) * tx[:, None]
        return xdq.astype(np.float64) @ wq.astype(np.float64).T * sw[:, None]
    return TRUE * tx[:, None] * sw[None, :]


def run_variant(model_dir, arrs, mode):
    indir = Path(model_dir).with_suffix(".in")
    indir.mkdir(exist_ok=True)
    for k, v in arrs.items():
        (indir / f"IN_{k}.bin").write_bytes(np.ascontiguousarray(v).tobytes())
    cmd = [str(DYNRUN), str(model_dir), str(indir), "--burst", "30"] + mode
    maclog = indir / "macmon.jsonl"
    mm = subprocess.Popen(["macmon", "pipe", "-i", "250"], stdout=open(maclog, "w"), stderr=subprocess.DEVNULL)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    mm.terminate()
    mm.wait(timeout=5)
    if proc.returncode != 0:
        return None, (proc.stderr or "")[-300:], None
    rec = json.loads(proc.stdout)
    y = np.fromfile(indir / "OUT_y.bin", dtype=np.float16 if "--cpu" not in mode else np.float16)
    ap = []
    for l in open(maclog):
        try:
            r = json.loads(l)
            r["_ts"] = datetime.fromisoformat(r["timestamp"]).timestamp()
            if t0 <= r["_ts"] <= time.time() and r.get("ane_power"):
                ap.append(r["ane_power"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return rec, None, (y, max(ap) if ap else 0.0)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    builders = {
        "mm_i8": build_mm_i8,
        "mm_f16": build_mm_f16,
        "conv_i8": build_conv_i8,
        "conv_dq": build_conv_dq,
        "baked": build_baked,
    }
    for name, b in builders.items():
        rec = {"variant": name}
        try:
            prog, ci = b()
            m = ct.convert(
                prog,
                convert_to="mlprogram",
                inputs=ci,
                minimum_deployment_target=TARGET,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )
            p = CACHE / f"{name}.mlpackage"
            if p.exists():
                import shutil

                shutil.rmtree(p)
            m.save(str(p))
            rec["convert"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec.update(convert="fail", err=str(e)[:200])
            emit(rec)
            continue
        arrs = reshape_for(name, inputs_for(name))
        for label, mode in (("ane_copy", []), ("ane_wrap", ["--wrap"]), ("cpu", ["--cpu"])):
            res, err, aux = run_variant(p, arrs, mode)
            if res is None:
                rec[label] = "fail"
                rec[label + "_err"] = err[:200] if err else "no output"
                continue
            y, peak = aux
            ref = ref_out(name).reshape(-1)
            if name in ("conv_i8", "conv_dq"):
                y = y.reshape(O, T).T  # conv output is (1, O, 1, T), O-major
            yy = y.astype(np.float64)
            n = min(len(yy), len(ref))
            cos = float(yy[:n] @ ref[:n] / (np.linalg.norm(yy[:n]) * np.linalg.norm(ref[:n]) + 1e-30))
            rec[label] = round(res["med_ms"], 3)
            rec[label + "_cos"] = round(cos, 7)
            if label == "ane_copy":
                rec["out_shape"] = list(y[: T * O].reshape(T, O).shape)
                rec["ane_peak_w"] = round(peak, 2)
        emit(rec)


if __name__ == "__main__":
    main()
