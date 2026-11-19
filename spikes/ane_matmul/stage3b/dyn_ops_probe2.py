"""3b-2 dynamic-weights probe, round 2: int8 weight INPUTS through cast-in-graph conv.

Ops discovery (round 1, results/dyn_ops_probe.jsonl): this coremltools rejects
int8 inputs to matmul (domain fp16/fp32/int32) and to conv (fp16/fp32), and
requires dequantize scale to be constexpr. The workable dynamic-weight MIL form
is therefore: int8 weight INPUT -> cast fp16 -> fp16 conv (ANE accumulates in
fp32) -> multiply by dynamic row scale tx and channel scale sw at the end.

Variants at production shapes (T=4608 rows, qkv site O=5760 I=3840):
  baked    current production constexpr_blockwise conv (control)
  cv_i8w   int8 weight input cast in-graph, activation fp16 input
  cv_f16w  fp16 weight input (no cast cost; doubles the handoff bytes)
  mm_f16   plain fp16 matmul with fp16 weight input (does ANE take matmul?)

Handoff modes per variant: copy (owned MLMultiArray + memcpy), wrap
(zero-copy pointer wrap - answers whether int8 CAN be wrapped; round-1's
rejection was the enum poison, not the wrap), cpu (CPUAndCPU reference).
Numbers vs fp64 emulation of the same math. macmon ane_power proves ANE use.
"""

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/ane3a/cache_dyn2")
RESULTS = HERE / "results"
DYNRUN = HERE.parent / "bench" / "dynrun"
TARGET = ct.target.iOS26 if hasattr(ct.target, "iOS26") else ct.target.iOS18

T, I, O = 4608, 3840, 5760
rng = np.random.default_rng(5)
x_true = (rng.standard_normal((T, I)) * 0.05).astype(np.float32)
w_true = (rng.standard_normal((O, I)) * 0.02).astype(np.float32)
tx = np.maximum(np.abs(x_true).max(axis=1) / 127.0, np.float32(1e-8)).astype(np.float32)
sw = np.maximum(np.abs(w_true).max(axis=1) / 127.0, np.float32(1e-8)).astype(np.float32)
xq = np.clip(np.round(x_true / tx[:, None]), -127, 127).astype(np.int8)
wq = np.clip(np.round(w_true / sw[:, None]), -127, 127).astype(np.int8)
TRUE = (
    (xq.astype(np.float32) @ wq.astype(np.float32).T).astype(np.float64)
    * tx.astype(np.float64)[:, None]
    * sw.astype(np.float64)[None, :]
)


def emit(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "dyn_ops_probe2.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def _conv_dyn(xspec_w_dtype):
    def build():
        specs = [
            mb.TensorSpec((T, I), dtype=types.fp16),
            mb.TensorSpec((O, I), dtype=types.int8 if xspec_w_dtype == "int8" else types.fp16),
            mb.TensorSpec((T,), dtype=types.fp16),
            mb.TensorSpec((O,), dtype=types.fp16),
        ]

        @mb.program(input_specs=specs, opset_version=TARGET)
        def prog(x, w, tx, sw):
            wd = mb.cast(x=w, dtype="fp16") if xspec_w_dtype == "int8" else w
            xcf = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, I, 1, -1])
            y = mb.conv(x=xcf, weight=mb.reshape(x=wd, shape=[O, I, 1, 1]), pad_type="valid")
            y2 = mb.transpose(x=mb.reshape(x=y, shape=[O, -1]), perm=[1, 0])
            return mb.mul(x=mb.mul(x=y2, y=mb.expand_dims(x=tx, axes=[1])), y=mb.reshape(x=sw, shape=[1, O]), name="y")

        return prog, [ct.TensorType(name=n) for n in ("x", "w", "tx", "sw")]

    return build


def build_baked():
    @mb.program(input_specs=[mb.TensorSpec((T, I), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        w4 = mb.constexpr_blockwise_shift_scale(
            data=np.ascontiguousarray(wq.reshape(O, I, 1, 1)), scale=sw.reshape(O, 1, 1, 1), offset=None
        )
        xcf = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, I, 1, -1])
        return mb.identity(
            x=mb.transpose(x=mb.reshape(x=mb.conv(x=xcf, weight=w4, pad_type="valid"), shape=[O, -1]), perm=[1, 0]),
            name="y",
        )

    return prog, [ct.TensorType(name="x", shape=[T, I])]


def build_mm_f16():
    @mb.program(
        input_specs=[mb.TensorSpec((T, I), dtype=types.fp16), mb.TensorSpec((O, I), dtype=types.fp16)],
        opset_version=TARGET,
    )
    def prog(x, w):
        return mb.matmul(x=x, y=mb.transpose(x=w, perm=[1, 0]), name="y")

    return prog, [ct.TensorType(name=n) for n in ("x", "w")]


def inputs_for(name):
    # x and w carry exact int8 codes as fp16 integers; the program applies tx/sw once
    xi = xq.astype(np.float16)
    if name == "baked":
        return {"x": (xq.astype(np.float32) * tx[:, None]).astype(np.float16)}
    if name == "mm_f16":
        return {
            "x": (xq.astype(np.float32) * tx[:, None]).astype(np.float16),
            "w": (wq.astype(np.float32) * sw[:, None]).astype(np.float16),
        }
    if name == "cv_f16w":
        return {"x": xi, "w": wq.astype(np.float16), "tx": tx.astype(np.float16), "sw": sw.astype(np.float16)}
    return {"x": xi, "w": wq, "tx": tx.astype(np.float16), "sw": sw.astype(np.float16)}


def run_mode(model_dir, arrs, mode):
    indir = Path(model_dir).with_suffix(".in")
    indir.mkdir(parents=True, exist_ok=True)
    for k, v in arrs.items():
        (indir / f"IN_{k}.bin").write_bytes(np.ascontiguousarray(v).tobytes())
    cmd = [str(DYNRUN), str(model_dir), str(indir), "--burst", "30"] + mode
    maclog = indir / "macmon.jsonl"
    mm = subprocess.Popen(["macmon", "pipe", "-i", "250"], stdout=open(maclog, "w"), stderr=subprocess.DEVNULL)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    mm.terminate()
    mm.wait(timeout=5)
    if proc.returncode != 0:
        return None, (proc.stderr or "")[-250:], None, None
    rec = json.loads(proc.stdout)
    y = np.fromfile(indir / "OUT_y.bin", dtype=np.float16)
    ap = []
    for l in open(maclog):
        try:
            r = json.loads(l)
            ts = datetime.fromisoformat(r["timestamp"]).timestamp()
            if t0 <= ts and r.get("ane_power"):
                ap.append(r["ane_power"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return rec, None, y, (max(ap) if ap else 0.0)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    builders = {"baked": build_baked, "cv_i8w": _conv_dyn("int8"), "cv_f16w": _conv_dyn("fp16"), "mm_f16": build_mm_f16}
    for name, b in builders.items():
        rec = {"variant": name, "shape": [T, O, I]}
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
                shutil.rmtree(p)
            m.save(str(p))
            rec["convert"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec.update(convert="fail", err=str(e)[:200])
            emit(rec)
            continue
        arrs = inputs_for(name)
        for label, mode in (("copy", []), ("wrap", ["--wrap"]), ("cpu", ["--cpu"])):
            res, err, y, peak = run_mode(p, arrs, mode)
            if res is None:
                rec[label] = "fail"
                rec[label + "_err"] = err
                continue
            if label in ("copy", "wrap"):
                got = y.astype(np.float64).reshape(TRUE.shape)
                cos = float(
                    (got.ravel() @ TRUE.ravel()) / (np.linalg.norm(got.ravel()) * np.linalg.norm(TRUE.ravel()) + 1e-30)
                )
                rec[label + "_ms"] = res["med_ms"]
                rec[label + "_cos"] = round(cos, 7)
                rec[label + "_anew"] = round(peak, 1)
                rec[label + "_tflops"] = round(2 * T * I * O / 1e12 / (res["med_ms"] / 1e3), 1)
            else:
                rec["cpu_ms"] = res["med_ms"]
        emit(rec)


if __name__ == "__main__":
    main()
