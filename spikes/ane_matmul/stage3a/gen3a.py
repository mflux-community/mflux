"""Stage 3a model builders: enumerated shapes, multifunction packages, group-wise int8.

All builders bake int8 weights as constants in a 1x1 Conv2d, channels-first
(1, C, 1, T), the V2 form from Stage 1. Weights are synthetic unless w_s is given.
"""

import coremltools as ct
import numpy as np
from coremltools.converters.mil import (
    Builder as mb,
    mil,
)
from coremltools.converters.mil.mil import get_new_symbol, types

TARGET = ct.target.iOS26 if hasattr(ct.target, "iOS26") else ct.target.iOS18


def _synth_weights(O, I, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.integers(-127, 128, size=(O, I, 1, 1), dtype=np.int8)
    s = (rng.random((O, 1, 1, 1)) * 0.01 + 0.001).astype(np.float16)
    return w, s


def _k_chunks(w, I, ksplit):
    if I % ksplit:
        raise ValueError(f"inner dimension {I} is not divisible by ksplit {ksplit}")
    kc = I // ksplit
    return kc, [np.ascontiguousarray(w[:, k * kc : (k + 1) * kc]) for k in range(ksplit)]


def _a8_dyn(x):
    """Per-token int8 activation quant/dequant inside the graph (the Stage 1 W8A8 form)."""
    amax = mb.reduce_max(x=mb.abs(x=x), axes=[1], keep_dims=True)
    sc = mb.maximum(x=mb.real_div(x=amax, y=np.float16(127.0)), y=np.float16(1e-4))
    q = mb.clip(x=mb.round(x=mb.real_div(x=x, y=sc)), alpha=np.float16(-127), beta=np.float16(127))
    return mb.mul(x=q, y=sc)


def _conv_body(x, ws_chunks, s, ksplit):
    acc = None
    for k in range(ksplit):
        if ksplit == 1:
            xs = x
        else:
            kc = ws_chunks[0].shape[1]
            xs = mb.slice_by_index(
                x=x,
                begin=[0, k * kc, 0, 0],
                end=[1, (k + 1) * kc, 1, 0],
                end_mask=[False, False, False, True],
            )
        ws = mb.constexpr_blockwise_shift_scale(data=ws_chunks[k], scale=s, offset=None)
        y = mb.conv(x=xs, weight=ws, pad_type="valid")
        acc = y if acc is None else mb.add(x=acc, y=y)
    return acc


def build_fixed(T, O, I, ksplit=1, a8=False, seed=0, w_s=None):
    """Fixed token count. w_s: optional (int8 w [O,I,1,1], fp16 s [O,1,1,1]) real/shared weights."""
    w, s = _synth_weights(O, I, seed) if w_s is None else w_s
    _, ws_chunks = _k_chunks(w, I, ksplit)

    @mb.program(input_specs=[mb.TensorSpec((1, I, 1, T), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        if a8:
            x = _a8_dyn(x)
        return mb.identity(x=_conv_body(x, ws_chunks, s, ksplit), name="y")

    return prog


def build_enum(O, I, token_list, ksplit=1, a8=False, seed=0, w_s=None):
    """One function, enumerated token counts on the width axis. Returns (program, ct_inputs)."""
    w, s = _synth_weights(O, I, seed) if w_s is None else w_s
    _, ws_chunks = _k_chunks(w, I, ksplit)
    tsym = get_new_symbol("t")

    @mb.program(input_specs=[mb.TensorSpec((1, I, 1, tsym), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        if a8:
            x = _a8_dyn(x)
        return mb.identity(x=_conv_body(x, ws_chunks, s, ksplit), name="y")

    ct_inputs = [
        ct.TensorType(
            name="x",
            shape=ct.EnumeratedShapes(
                shapes=[[1, I, 1, t] for t in token_list],
                default=[1, I, 1, token_list[0]],
            ),
        )
    ]
    return prog, ct_inputs


def build_block_programs(T, layers, a8_both=True, seed=0):
    """Multifunction program with one W8A16 function per layer, plus a W8A8 twin that shares the same weight arrays.

    layers: dict name -> (O, I, ksplit). Returns (program, {fn_name: (O, I)}).
    """
    prog = mil.Program()
    shapes = {}
    first = None
    for i, (name, (O, I, ksplit)) in enumerate(layers.items()):
        w, s = _synth_weights(O, I, seed + i)
        _, ws_chunks = _k_chunks(w, I, ksplit)

        def make(a8, ws_chunks=ws_chunks, s=s, I=I, T=T, ksplit=ksplit):
            @mb.function(input_specs=[mb.TensorSpec((1, I, 1, T), dtype=types.fp16)], opset_version=TARGET)
            def f(x):
                if a8:
                    x = _a8_dyn(x)
                return mb.identity(x=_conv_body(x, ws_chunks, s, ksplit), name="y")

            return f

        prog.add_function(name, make(False))
        shapes[name] = (O, I)
        if first is None:
            first = name
        if a8_both:
            prog.add_function(name + "_a8", make(True))
            shapes[name + "_a8"] = (O, I)
    prog.default_function_name = first
    prog.export_as_multifunction = True
    return prog, shapes


def build_blockwise(T, O, I, q, scale, offset, ksplit=1):
    """Group-wise int8 conv: dequant is scale * (data - offset) per group of `block` along I.

    q: int8/uint8 [O, I, 1, 1]. scale, offset: [O, I//block, 1, 1] fp16 or fp32.
    """
    _, qs = _k_chunks(np.ascontiguousarray(q), I, ksplit)
    if ksplit > 1:
        gc = scale.shape[1] // ksplit
        ss = [np.ascontiguousarray(scale[:, k * gc : (k + 1) * gc]) for k in range(ksplit)]
        oo = None if offset is None else [np.ascontiguousarray(offset[:, k * gc : (k + 1) * gc]) for k in range(ksplit)]
    else:
        ss = [scale]
        oo = [offset]

    @mb.program(input_specs=[mb.TensorSpec((1, I, 1, T), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        acc = None
        for k in range(ksplit):
            if ksplit == 1:
                xs = x
            else:
                kc = qs[k].shape[1]
                xs = mb.slice_by_index(
                    x=x,
                    begin=[0, k * kc, 0, 0],
                    end=[1, (k + 1) * kc, 1, 0],
                    end_mask=[False, False, False, True],
                )
            ws = mb.constexpr_blockwise_shift_scale(data=qs[k], scale=ss[k], offset=None if oo[k] is None else oo[k])
            y = mb.conv(x=xs, weight=ws, pad_type="valid")
            acc = y if acc is None else mb.add(x=acc, y=y)
        return mb.identity(x=acc, name="y")

    return prog


def convert(prog, out_path, inputs=None):
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=inputs,
        minimum_deployment_target=TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    m.save(str(out_path))
    return out_path
