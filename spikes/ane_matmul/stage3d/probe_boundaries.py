"""A: localise where the ANE diverges from the intended math, inside the fused MLP program.

Boundaries: a = xW1^T, b = xW3^T, h = silu(a)*(b/32), y = hW2^T.

Each boundary gets its own program that declares ONLY the inputs it consumes. Declaring an input
the returned branch does not use makes coremltools insert a placeholder identity on the int8 weight
input, and Core ML rejects int8 identity ("Param 'x' has incorrect type for operator 'identity'"),
so a full-graph-minus-one-output variant will not compile.

Run through the bridge, which is the path production uses -- coremltools' own predict() cannot take
int8 graph inputs at all (RuntimeError: value type not convertible).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # stage3d
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ane_linear"))

import _ane_linear as ext  # noqa: E402
import coremltools as ct  # noqa: E402
import mlx.core as mx  # noqa: E402,F401  -- must precede _ane_linear: it is built with NB_DOMAIN mlx
from coremltools.converters.mil import Builder as mb  # noqa: E402
from coremltools.converters.mil.mil import types  # noqa: E402
from mlp_program import (  # noqa: E402
    DIM,
    HEADROOM,
    HIDDEN,
    INPUT_SHRINK,
    MlpSpec,
    program_scales,
    quantize_per_row,
)

M = 1024
CACHE = HERE / "cache" / "probeA"
COMPILED = Path.home() / "Library/Caches/mflux/ane_wl3d_probeA"
CACHE.mkdir(parents=True, exist_ok=True)
COMPILED.mkdir(parents=True, exist_ok=True)
D, H = DIM, HIDDEN


def _deq(w, scale, spec, name):
    """int8 -> fp16 widening, plus the per-row scale when the scale lives in the program."""
    wf = mb.dequantize(input=w, scale=np.float16(1.0), name=f"{name}_wf") if spec.weight_dtype == "int8" else w
    if spec.scale_mode == "weight":
        wf = mb.mul(x=wf, y=mb.reshape(x=scale, shape=(wf.shape[0], 1)), name=f"{name}_wdeq")
    return wf


def _mm(x, wf, k_chunks, in_dim, out_dim, name):
    step = in_dim // k_chunks
    acc = None
    for i in range(k_chunks):
        lo, hi = i * step, (i + 1) * step
        xi = x if k_chunks == 1 else mb.slice_by_index(x=x, begin=[0, 0, lo], end=[1, M, hi], name=f"{name}_x{i}")
        wi = wf if k_chunks == 1 else mb.slice_by_index(x=wf, begin=[0, lo], end=[out_dim, hi], name=f"{name}_w{i}")
        yi = mb.matmul(x=xi, y=wi, transpose_y=True, name=f"{name}_mm{i}")
        acc = yi if acc is None else mb.add(x=acc, y=yi, name=f"{name}_acc{i}")
    return acc


def _oscale(acc, scale, spec):
    return mb.mul(x=acc, y=scale, name="oscale") if spec.scale_mode == "output" else acc


def build(spec, which):
    """One program per boundary, declaring only the inputs that boundary uses."""
    i8 = types.int8 if spec.weight_dtype == "int8" else types.fp16
    opset = ct.target.iOS26 if spec.weight_dtype == "int8" else ct.target.iOS18

    def shr(x):
        return mb.mul(x=x, y=np.float16(1.0 / INPUT_SHRINK), name="x_shrink") if spec.scale_mode == "output" else x

    if which == "a":
        specs = [
            mb.TensorSpec(shape=(1, M, D), dtype=types.fp16),
            mb.TensorSpec(shape=(H, D), dtype=i8),
            mb.TensorSpec(shape=(H,), dtype=types.fp16),
        ]

        @mb.program(input_specs=specs, opset_version=opset)
        def prog(x_rows, w1, s1):
            return mb.identity(
                x=_oscale(_mm(shr(x_rows), _deq(w1, s1, spec, "w1"), spec.k13_chunks, D, H, "w1"), s1, spec),
                name="y_rows",
            )

        cols, names, shapes, int8s = H, ["x_rows", "w1", "s1"], [[1, M, D], [H, D], [H]], [0, 1, 0]

    elif which == "b":
        specs = [
            mb.TensorSpec(shape=(1, M, D), dtype=types.fp16),
            mb.TensorSpec(shape=(H, D), dtype=i8),
            mb.TensorSpec(shape=(H,), dtype=types.fp16),
        ]

        @mb.program(input_specs=specs, opset_version=opset)
        def prog(x_rows, w3, s3):
            b = _oscale(_mm(shr(x_rows), _deq(w3, s3, spec, "w3"), spec.k13_chunks, D, H, "w3"), s3, spec)
            return mb.identity(x=mb.mul(x=b, y=np.float16(1.0 / HEADROOM), name="b32"), name="y_rows")

        cols, names, shapes, int8s = H, ["x_rows", "w3", "s3"], [[1, M, D], [H, D], [H]], [0, 1, 0]

    elif which == "h":
        specs = [
            mb.TensorSpec(shape=(1, M, D), dtype=types.fp16),
            mb.TensorSpec(shape=(H, D), dtype=i8),
            mb.TensorSpec(shape=(H, D), dtype=i8),
            mb.TensorSpec(shape=(H,), dtype=types.fp16),
            mb.TensorSpec(shape=(H,), dtype=types.fp16),
        ]

        @mb.program(input_specs=specs, opset_version=opset)
        def prog(x_rows, w1, w3, s1, s3):
            xs = shr(x_rows)
            a = _oscale(_mm(xs, _deq(w1, s1, spec, "w1"), spec.k13_chunks, D, H, "w1"), s1, spec)
            b = _oscale(_mm(xs, _deq(w3, s3, spec, "w3"), spec.k13_chunks, D, H, "w3"), s3, spec)
            b32 = mb.mul(x=b, y=np.float16(1.0 / HEADROOM), name="b32")
            return mb.identity(x=mb.mul(x=mb.silu(x=a, name="silu"), y=b32, name="swiglu"), name="y_rows")

        cols, names, shapes, int8s = (
            H,
            ["x_rows", "w1", "w3", "s1", "s3"],
            [[1, M, D], [H, D], [H, D], [H], [H]],
            [0, 1, 1, 0, 0],
        )

    elif which == "y":
        specs = [
            mb.TensorSpec(shape=(1, M, D), dtype=types.fp16),
            mb.TensorSpec(shape=(H, D), dtype=i8),
            mb.TensorSpec(shape=(H, D), dtype=i8),
            mb.TensorSpec(shape=(D, H), dtype=i8),
            mb.TensorSpec(shape=(H,), dtype=types.fp16),
            mb.TensorSpec(shape=(H,), dtype=types.fp16),
            mb.TensorSpec(shape=(D,), dtype=types.fp16),
        ]

        @mb.program(input_specs=specs, opset_version=opset)
        def prog(x_rows, w1, w3, w2, s1, s3, s2):
            xs = shr(x_rows)
            a = _oscale(_mm(xs, _deq(w1, s1, spec, "w1"), spec.k13_chunks, D, H, "w1"), s1, spec)
            b = _oscale(_mm(xs, _deq(w3, s3, spec, "w3"), spec.k13_chunks, D, H, "w3"), s3, spec)
            h = mb.mul(
                x=mb.silu(x=a, name="silu"), y=mb.mul(x=b, y=np.float16(1.0 / HEADROOM), name="b32"), name="swiglu"
            )
            # The w2 scale must be applied here too: weight mode folds it in _deq, output mode
            # applies it to the matmul result. Leaving it out made this arm 1/s2 too large.
            y = _oscale(_mm(h, _deq(w2, s2, spec, "w2"), spec.k2_chunks, H, D, "w2"), s2, spec)
            return mb.identity(x=y, name="y_rows")

        cols, names, shapes, int8s = (
            D,
            ["x_rows", "w1", "w3", "w2", "s1", "s3", "s2"],
            [[1, M, D], [H, D], [H, D], [D, H], [H], [H], [D]],
            [0, 1, 1, 1, 0, 0, 0],
        )
    else:
        raise ValueError(which)

    # Every declared input is consumed in both modes: weight mode folds the scale inside _deq,
    # output mode applies it to the matmul result in _oscale.
    model = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=opset,
    )
    return model, names, shapes, int8s, cols


def run(model, names, shapes, int8s, cols, feed, tag):
    pkg = CACHE / f"{tag}.mlpackage"
    if pkg.exists():
        import shutil

        shutil.rmtree(pkg)
    model.save(str(pkg))
    mc = ext.wl_fused_load(str(pkg), str(COMPILED / tag), names, shapes, int8s, "y_rows", [1, M, cols])
    out = np.zeros((1, M, cols), dtype=np.float16)
    ins = [np.ascontiguousarray(feed[n]) for n in names]
    ext.wl_fused_predict(mc, [a.ctypes.data for a in ins], out.ctypes.data)
    if e := mc.last_error():
        raise RuntimeError(e)
    cb = mc.copybacks()
    del mc
    return out, cb


def cmp(tag, y, r):
    y = np.asarray(y, dtype=np.float64).ravel()
    r = np.asarray(r, dtype=np.float64).ravel()
    cos = float(r @ y / (np.linalg.norm(r) * np.linalg.norm(y)))
    rel = float(np.linalg.norm(y - r) / np.linalg.norm(r))
    gain = float(r @ y / (y @ y))
    print(f"    {tag:>4}  cos={cos:.7f}  relerr={rel:.5f}  gain={gain:.5f}  peak={np.abs(y).max():.4f}")
    return cos


rng = np.random.default_rng(0)
init = np.random.default_rng(7)
w1 = init.normal(0, 0.016, (H, D)).astype(np.float32)
w3 = init.normal(0, 0.016, (H, D)).astype(np.float32)
w2 = init.normal(0, 0.016, (D, H)).astype(np.float32)
x = rng.normal(0, 1.0, (1, M, D)).astype(np.float16)

for scale_mode in ("weight", "output"):
    spec = MlpSpec(mpad=M, scale_mode=scale_mode)
    (q1, s1), (q3, s3), (q2, s2) = quantize_per_row(w1), quantize_per_row(w3), quantize_per_row(w2)
    sc = program_scales(spec, s1, s3, s2)
    feed = {"x_rows": x, "w1": q1, "w3": q3, "w2": q2, "s1": sc["s1"], "s3": sc["s3"], "s2": sc["s2"]}

    xf = x.astype(np.float32)
    a_ref = xf @ w1.T
    b_ref = xf @ w3.T
    h_ref = (a_ref / (1.0 + np.exp(-a_ref))) * (b_ref / HEADROOM)
    # The program returns y/32: h_ref is already h_true/32, so this must NOT divide again.
    y_ref = h_ref @ w2.T
    refs = {"a": a_ref, "b": b_ref / HEADROOM, "h": h_ref, "y": y_ref}

    print(f"\n=== scale_mode={scale_mode}  Mpad={M}  k13={spec.k13_chunks} ===")
    print("    ANE boundary vs exact fp32 of that same boundary:")
    for which in ("a", "b", "h"):
        model, names, shapes, int8s, cols = build(spec, which)
        out, cb = run(model, names, shapes, int8s, cols, feed, f"{scale_mode}_{which}")
        cmp(which, out, refs[which])
        print(f"         copybacks={cb}")
    for k2 in (1, 2, 4):
        spec2 = MlpSpec(mpad=M, scale_mode=scale_mode, k2_chunks=k2)
        model, names, shapes, int8s, cols = build(spec2, "y")
        out, cb = run(model, names, shapes, int8s, cols, feed, f"{scale_mode}_y_k2{k2}")
        cmp(f"y_k2={k2}", out, y_ref)
        print(f"         copybacks={cb}")
