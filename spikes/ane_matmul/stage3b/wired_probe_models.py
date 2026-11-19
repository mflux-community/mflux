"""3b-2 wired-reduction probe: builders for fixed-shape and fused-pair models.

Two questions, one matrix:
  - does EnumeratedShapes add wired memory versus one fixed shape?
  - does ONE model with two outputs (pair A = qkv + w1w3, pair B = o + w2)
    wire less than two separate models? The block dataflow needs the SDPA and
    the silu between the pair halves, so two live models per block is the
    minimum instance count short of porting attention itself to Core ML.

Boundaries mirror production: w8a16 takes (rows, I) fp16 token-major; w8a8
takes channels-first (1, I, 1, rows) int8 + (1, 1, 1, rows) fp16 tok. Pair
programs return named outputs y1, y2. Fixed shapes only (rows=4608): wired
scales with the shapes actually predicted, so fixed arms isolate the enum tax.
"""

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ane_models as AM  # noqa: E402

TARGET = AM.TARGET
ROWS = 4608


def _conv_branch(xcf, wq, sc, ksplit):
    O, I = wq.shape
    w4 = wq.reshape(O, I, 1, 1)
    s4 = sc.reshape(O, 1, 1, 1)
    kc, ws = AM._chunks(w4, I, ksplit)
    acc = None
    for k in range(ksplit):
        xs = xcf if ksplit == 1 else AM._slice_width(xcf, k, kc)
        wf = mb.constexpr_blockwise_shift_scale(data=ws[k], scale=s4, offset=None)
        y = mb.conv(x=xs, weight=wf, pad_type="valid")
        acc = y if acc is None else mb.add(x=acc, y=y)
    return acc


def _fp16_branch(xcf, wq, sc, O, ksplit=1):
    return AM._to_token_major(_conv_branch(xcf, wq, sc, ksplit), O)


def _q8_branch(xcf, tok, wq, sc, O, ksplit=1):
    xd = mb.dequantize(input=xcf, scale=np.float16(1.0))
    acc = _conv_branch(xd, wq, sc, ksplit)
    return AM._to_token_major(mb.mul(x=acc, y=tok), O)


def build_fixed(wq, sc, prec, ksplit=1, rows=ROWS):
    """Single concrete-shape replacement of the enumerated production builders."""
    O, I = wq.shape
    if prec == "w8a16":

        @mb.program(input_specs=[mb.TensorSpec((rows, I), dtype=types.fp16)], opset_version=TARGET)
        def prog(x):
            return mb.identity(x=_fp16_branch(AM._to_channels_first(x, I), wq, sc, O, ksplit), name="y")

        ct_in = [ct.TensorType(name="x", shape=[rows, I])]
    else:

        @mb.program(
            input_specs=[
                mb.TensorSpec((1, I, 1, rows), dtype=types.int8),
                mb.TensorSpec((1, 1, 1, rows), dtype=types.fp16),
            ],
            opset_version=TARGET,
        )
        def prog(x, tok):
            return mb.identity(x=_q8_branch(x, tok, wq, sc, O, ksplit), name="y")

        ct_in = [ct.TensorType(name="x", shape=[1, I, 1, rows]), ct.TensorType(name="tok", shape=[1, 1, 1, rows])]
    return prog, ct_in


def build_pair_a(wq1, sc1, wq2, sc2, prec, rows=ROWS):
    """M_A: qkv prefix and w1w3 prefix share input x, so they fuse into ONE wider conv.

    Output is a single (rows, O1+O2) block y; the caller slices columns. This is
    the point of the pair: two model instances collapse into one, one predict.
    """
    wq = np.ascontiguousarray(np.concatenate([wq1, wq2], axis=0))
    sc = np.ascontiguousarray(np.concatenate([sc1, sc2], axis=0))
    return build_fixed(wq, sc, prec, ksplit=1, rows=rows)


def build_pair_b(wq1, sc1, wq2, sc2, prec, ksplit2=4, rows=ROWS):
    """M_B: the o prefix (input a) and the w2 prefix (input h) cannot share a conv, so
    this is one model, two inputs, two branches concatenated on axis 1 into one y.

    Still one predict call and one model instance instead of two.
    """
    O1, I1 = wq1.shape
    O2, I2 = wq2.shape
    if prec == "w8a16":

        @mb.program(
            input_specs=[mb.TensorSpec((rows, I1), dtype=types.fp16), mb.TensorSpec((rows, I2), dtype=types.fp16)],
            opset_version=TARGET,
        )
        def prog(a, h):
            y1 = _fp16_branch(AM._to_channels_first(a, I1), wq1, sc1, O1)
            y2 = _fp16_branch(AM._to_channels_first(h, I2), wq2, sc2, O2, ksplit2)
            return mb.concat(values=[y1, y2], axis=1, name="y")

        ct_in = [ct.TensorType(name="a", shape=[rows, I1]), ct.TensorType(name="h", shape=[rows, I2])]
    else:

        @mb.program(
            input_specs=[
                mb.TensorSpec((1, I1, 1, rows), dtype=types.int8),
                mb.TensorSpec((1, 1, 1, rows), dtype=types.fp16),
                mb.TensorSpec((1, I2, 1, rows), dtype=types.int8),
                mb.TensorSpec((1, 1, 1, rows), dtype=types.fp16),
            ],
            opset_version=TARGET,
        )
        def prog(aq, atok, hq, htok):
            y1 = _q8_branch(aq, atok, wq1, sc1, O1)
            y2 = _q8_branch(hq, htok, wq2, sc2, O2, ksplit2)
            return mb.concat(values=[y1, y2], axis=1, name="y")

        ct_in = [
            ct.TensorType(name="aq", shape=[1, I1, 1, rows]),
            ct.TensorType(name="atok", shape=[1, 1, 1, rows]),
            ct.TensorType(name="hq", shape=[1, I2, 1, rows]),
            ct.TensorType(name="htok", shape=[1, 1, 1, rows]),
        ]
    return prog, ct_in


def convert_multi(prog, ct_in, out_path, n_out):
    outs = [ct.TensorType(name=f"y{i + 1}") for i in range(n_out)]
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=ct_in,
        outputs=outs,
        minimum_deployment_target=TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return out_path
