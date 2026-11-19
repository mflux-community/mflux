"""Enumerated-tier ANE model builders for the 3b-1 extension.

Both models take the token-major interface the bridge wraps directly:
  build_enum_w8a16: x fp16 (t, I) -> y fp16 (t, O), baked per-channel int8
    weights, K-split chunks summed.
  build_enum_q8pre: x int8 (t, I), tok fp16 (t,) per-token scales applied
    AFTER the matmul (Stage 3a test 7 design rule) -> y fp16 (t, O).
The channels-first transpose runs inside the model (the Stage 1/2 'to' form)
with dynamic-shape machinery so one function covers all enumerated tiers.
"""

import itertools

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types

TARGET = ct.target.iOS26 if hasattr(ct.target, "iOS26") else ct.target.iOS18
TIERS = [4608, 6528, 8320, 16128]
_SYM = itertools.count()  # coremltools symbol names are global: one per build


def _new_sym(prefix):
    return get_new_symbol(f"{prefix}{next(_SYM)}")


def _chunks(w4, I, ksplit):
    if I % ksplit:
        raise ValueError(f"inner dim {I} not divisible by ksplit {ksplit}")
    kc = I // ksplit
    return kc, [np.ascontiguousarray(w4[:, k * kc : (k + 1) * kc]) for k in range(ksplit)]


def _slice_width(xs, k, kc):
    return mb.slice_by_index(
        x=xs,
        begin=[0, k * kc, 0, 0],
        end=[1, (k + 1) * kc, 1, 0],
        end_mask=[False, False, False, True],
    )


def _to_channels_first(x2d, I):
    """(t, I) -> (1, I, 1, t) with a dynamic t (reshape -1 wildcard)."""
    xt = mb.transpose(x=x2d, perm=[1, 0])
    return mb.reshape(x=xt, shape=[1, I, 1, -1])


def _to_token_major(acc, O):
    """(1, O, 1, t) -> (t, O) with a dynamic t (reshape -1 wildcard)."""
    flat = mb.reshape(x=acc, shape=[O, -1])
    return mb.transpose(x=flat, perm=[1, 0])


def quantize_perchannel(w):
    """w fp32 [O, I] -> (int8 [O,I], fp16 [O]) symmetric per output channel."""
    sc = (np.abs(w).max(axis=1) / 127.0).astype(np.float32)
    sc = np.maximum(sc, 1e-12)
    wq = np.clip(np.round(w / sc[:, None]), -127, 127).astype(np.int8)
    return wq, sc.astype(np.float16)


def _enum_inputs(specs, tiers):
    out = []
    for name, mk in specs:
        shapes = [mk(t) for t in tiers]
        out.append(ct.TensorType(name=name, shape=ct.EnumeratedShapes(shapes=shapes, default=shapes[0])))
    return out


def build_enum_w8a16(wq, sc, ksplit=1, tiers=None):
    tiers = tiers or TIERS
    O, I = wq.shape
    w4 = wq.reshape(O, I, 1, 1)
    s4 = sc.reshape(O, 1, 1, 1)
    kc, ws = _chunks(w4, I, ksplit)
    tsym = _new_sym("t")

    @mb.program(input_specs=[mb.TensorSpec((tsym, I), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        xcf = _to_channels_first(x, I)
        acc = None
        for k in range(ksplit):
            xs = xcf if ksplit == 1 else _slice_width(xcf, k, kc)
            wf = mb.constexpr_blockwise_shift_scale(data=ws[k], scale=s4, offset=None)
            y = mb.conv(x=xs, weight=wf, pad_type="valid")
            acc = y if acc is None else mb.add(x=acc, y=y)
        return mb.identity(x=_to_token_major(acc, O), name="y")

    return prog, _enum_inputs([("x", lambda t: [t, I])], tiers)


def build_enum_q8pre(wq, sc, ksplit=1, tiers=None):
    """W8A8 pre form: int8 x (1, I, 1, t), fp16 tok (1, 1, 1, t) -> y fp16 (t, O).

    The boundary MUST be channels-first with int8: an enumerated int8 input in
    token-major layout fails the CoreML plan builder
    ("Invalid multi-array data type: 00020008", stage3b/diag_q8plan.py arms
    C vs D). The bridge transposes (t, I) -> (1, I, 1, t) on the host copy.
    """
    tiers = tiers or TIERS
    O, I = wq.shape
    w4 = wq.reshape(O, I, 1, 1)
    s4 = sc.reshape(O, 1, 1, 1)
    kc, ws = _chunks(w4, I, ksplit)
    tsym = _new_sym("tq8")

    @mb.program(
        input_specs=[
            mb.TensorSpec((1, I, 1, tsym), dtype=types.int8),
            mb.TensorSpec((1, 1, 1, tsym), dtype=types.fp16),
        ],
        opset_version=TARGET,
    )
    def prog(x, tok):
        acc = None
        for k in range(ksplit):
            xs = x if ksplit == 1 else _slice_width(x, k, kc)
            xs = mb.dequantize(input=xs, scale=np.float16(1.0))
            wf = mb.constexpr_blockwise_shift_scale(data=ws[k], scale=s4, offset=None)
            y = mb.conv(x=xs, weight=wf, pad_type="valid")
            acc = y if acc is None else mb.add(x=acc, y=y)
        y = mb.mul(x=acc, y=tok)
        return mb.identity(x=_to_token_major(y, O), name="y")

    return prog, _enum_inputs([("x", lambda t: [1, I, 1, t]), ("tok", lambda t: [1, 1, 1, t])], tiers)


def convert(prog, ct_inputs, out_path):
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=ct_inputs,
        minimum_deployment_target=TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    m.save(str(out_path))
    return out_path


def pick_tier(rows, tiers=None):
    tiers = tiers or TIERS
    for t in tiers:
        if rows <= t:
            return t
    raise ValueError(f"rows {rows} exceeds largest tier {tiers[-1]}")
