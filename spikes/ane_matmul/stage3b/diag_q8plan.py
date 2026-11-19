"""Bisect the q8pre 'Error in building plan': which structure breaks CoreML.

A: fixed T, token-major int8 boundary (T, I), tok (T,), in-graph int8 transpose.
B: fixed T, channels-first int8 boundary (1, I, 1, T), tok (1, 1, 1, T).
C: B with EnumeratedShapes tiers.
Prediction runs through an embedded Swift probe (same MLMultiArray path the
C++ bridge uses, since coremltools' Python predict cannot feed int8 arrays).
Every model outputs token-major (T, O) fp16.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ane_models as AM

TARGET = AM.TARGET
CACHE = Path("/tmp/ane3a/cache_diag")
O_ANE, I, T = 1920, 3840, 4608
TIERS = [4608, 6528]

PROBE = r"""
import CoreML
import Foundation
let a = CommandLine.arguments   // MODEL OUT.bin X.bin TOK.bin
let url = try MLModel.compileModel(at: URL(fileURLWithPath: a[1]))
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
let model = try MLModel(contentsOf: url, configuration: cfg)
var feats: [String: MLFeatureValue] = [:]
for (k, n) in ["x", "tok"].enumerated() {
    let d = model.modelDescription.inputDescriptionsByName[n]!.multiArrayConstraint!
    let data = try Data(contentsOf: URL(fileURLWithPath: a[3 + k]))
    let esize = d.dataType == .float16 ? 2 : 1
    var shape = d.shape.map { $0.intValue }
    let prefixCount = shape.dropLast().reduce(1, *)
    shape[shape.count - 1] = data.count / esize / prefixCount
    let m = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: d.dataType)
    _ = data.withUnsafeBytes { src in memcpy(m.dataPointer, src.baseAddress!, data.count) }
    feats[n] = MLFeatureValue(multiArray: m)
}
let out0 = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feats)).featureValue(for: "y")!.multiArrayValue!
var out = out0
if a.count > 5 && a[5] == "backings" {
    let shp = out0.shape.map { $0.intValue }
    let backing = try MLMultiArray(dataPointer: UnsafeMutableRawPointer(mutating: malloc(out0.count * 2)),
                                   shape: shp.map { NSNumber(value: $0) }, dataType: .float16,
                                   strides: [NSNumber(value: shp[1]), NSNumber(value: 1)], deallocator: nil)
    let opts = MLPredictionOptions(); opts.outputBackings = ["y": backing]
    let r = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feats), options: opts)
    out = r.featureValue(for: "y")!.multiArrayValue!
    print("{\"backing_used\": \(out.dataPointer == backing.dataPointer)}")
}
print("{\"shape\": \(out.shape.map { $0.intValue })}")
let n = out.count
var buf = [Float16](repeating: 0, count: n)
let p = out.dataPointer.bindMemory(to: Float16.self, capacity: n)
let st = out.strides.map { $0.intValue }, sh = out.shape.map { $0.intValue }
let R = sh[sh.count - 2], C = sh[sh.count - 1]
let r0 = st[st.count - 2], c0 = st[st.count - 1]
for r in 0..<R { for c in 0..<C { buf[r * C + c] = p[r * r0 + c * c0] } }
try Data(bytes: buf, count: n * 2).write(to: URL(fileURLWithPath: a[2]))
"""


def build_a():
    tsym = get_new_symbol("ta")

    @mb.program(
        input_specs=[mb.TensorSpec((tsym, I), dtype=types.int8), mb.TensorSpec((tsym,), dtype=types.fp16)],
        opset_version=TARGET,
    )
    def prog(x, tok):
        xcf = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, I, 1, -1])
        xd = mb.dequantize(input=xcf, scale=np.float16(1.0))
        wf = mb.constexpr_blockwise_shift_scale(data=WS, scale=S4, offset=None)
        acc = mb.conv(x=xd, weight=wf, pad_type="valid")
        y2 = mb.transpose(x=mb.reshape(x=acc, shape=[O_ANE, -1]), perm=[1, 0])
        return mb.mul(x=y2, y=mb.expand_dims(x=tok, axes=[1]), name="y")

    return prog, [ct.TensorType(name="x", shape=[T, I]), ct.TensorType(name="tok", shape=[T])]


def build_c(enum=False):
    def mk_prog(tdim):
        tsym = get_new_symbol("tce" if enum else "tcf")

        @mb.program(
            input_specs=[
                mb.TensorSpec((1, I, 1, tsym if enum else tdim), dtype=types.int8),
                mb.TensorSpec((1, 1, 1, tsym if enum else tdim), dtype=types.fp16),
            ],
            opset_version=TARGET,
        )
        def prog(x, tok):
            xd = mb.dequantize(input=x, scale=np.float16(1.0))
            wf = mb.constexpr_blockwise_shift_scale(data=WS, scale=S4, offset=None)
            acc = mb.conv(x=xd, weight=wf, pad_type="valid")
            y = mb.mul(x=acc, y=tok)
            y2 = mb.transpose(x=mb.reshape(x=y, shape=[O_ANE, -1]), perm=[1, 0])
            return mb.identity(x=y2, name="y")

        return prog

    if not enum:
        return mk_prog(T), None
    prog = mk_prog(None)
    xt = ct.TensorType(
        name="x", shape=ct.EnumeratedShapes(shapes=[[1, I, 1, t] for t in TIERS], default=[1, I, 1, TIERS[0]])
    )
    tt = ct.TensorType(
        name="tok", shape=ct.EnumeratedShapes(shapes=[[1, 1, 1, t] for t in TIERS], default=[1, 1, 1, TIERS[0]])
    )
    return prog, [xt, tt]


def build_d():
    """Current failing production design: token-major int8 + EnumeratedShapes + 1-D tok."""
    return AM.build_enum_q8pre(WQ, SC, ksplit=1, tiers=TIERS)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    probe_src = CACHE / "probe.swift"
    probe_bin = CACHE / "probe"
    probe_src.write_text(PROBE)
    subprocess.run(["swiftc", "-O", str(probe_src), "-o", str(probe_bin)], check=True)

    # Constant inputs: x = all 3 (int8), tok = all 0.02 (fp16). Output (T, O) row-major.
    x_bin = CACHE / "x.bin"
    x_bin.write_bytes(np.full((T, I), 3, dtype=np.int8).tobytes())
    tok_bin = CACHE / "tok.bin"
    tok_bin.write_bytes(np.full((T,), 0.02, dtype=np.float16).tobytes())
    ref = (3.0 * 0.02) * (WQ.astype(np.float64) * SC.astype(np.float64)[:, None]).sum(axis=1)

    for name, prog, inputs, backings, reuse in [
        ("a_tokmajor_fixed", *build_a(), False, None),
        ("b_chanfirst_fixed", *build_c(enum=False), False, None),
        ("c_chanfirst_enum", *build_c(enum=True), False, None),
        ("d_tokmajor_enum", *build_d(), False, None),
        ("e_d_with_backings", None, None, True, "d_tokmajor_enum"),
    ]:
        path = CACHE / f"{name}.mlpackage"
        if prog is not None:
            try:
                m = ct.convert(
                    prog,
                    convert_to="mlprogram",
                    inputs=inputs,
                    minimum_deployment_target=TARGET,
                    compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                )
                m.save(str(path))
            except Exception as e:  # noqa: BLE001
                print(f"ARM {name}: CONVERT FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
        elif reuse is not None:
            path = CACHE / f"{reuse}.mlpackage"
        y_path = CACHE / f"{name}_y.bin"
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "err.log"
            cmd = [str(probe_bin), str(path), str(y_path), str(x_bin), str(tok_bin)]
            if backings:
                cmd.append("backings")
            with open(log, "wb") as f:
                rc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=f, timeout=300)
            errtxt = log.read_text()[:300].replace("\n", " ")
        if rc.returncode != 0:
            print(f"ARM {name}: PREDICT FAIL rc={rc.returncode} {errtxt}", flush=True)
            continue
        y = np.fromfile(y_path, dtype=np.float16).astype(np.float64).reshape(T, O_ANE)
        rel = float(np.max(np.abs(y - ref[None, :])) / np.abs(ref).max())
        print(f"ARM {name}: predict OK {rc.stdout.decode().strip()} max_rel_vs_ref={rel:.6f}", flush=True)


rng = np.random.default_rng(7)
w = (rng.standard_normal((O_ANE, I)) * 0.02).astype(np.float32)
WQ, SC = AM.quantize_perchannel(w)
WS = WQ.reshape(O_ANE, I, 1, 1)
S4 = SC.reshape(O_ANE, 1, 1, 1)

if __name__ == "__main__":
    main()
