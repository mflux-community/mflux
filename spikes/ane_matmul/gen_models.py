"""Generate Core ML matmul models for the ANE spike.

V1: W (int8) and scale (fp16) are inputs.  V2: W and scale are constants.
Usage: gen_models.py OUTDIR TOKENS OUT IN [--variant v1|v2|v2k2|v2k4|v2a8|v2a8k2|c1|c2|c4|c1s|c4s|c1d|c4d|c1p|c4p|t1|t4]
"""

import argparse
import os

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

TARGET = ct.target.iOS26 if hasattr(ct.target, "iOS26") else ct.target.iOS18


def build_v1(T, O, I):
    @mb.program(
        input_specs=[
            mb.TensorSpec((T, I), dtype=types.fp16),
            mb.TensorSpec((O, I), dtype=types.int8),
            mb.TensorSpec((O,), dtype=types.fp16),
        ],
        opset_version=TARGET,
    )
    def prog(x, w, s):
        wf = mb.cast(x=w, dtype="fp16")
        y = mb.matmul(x=x, y=wf, transpose_y=True)
        return mb.mul(x=y, y=s, name="y")

    return prog


def build_v2(T, O, I, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.integers(-127, 128, size=(O, I), dtype=np.int8)
    s = (rng.random((O, 1)) * 0.01 + 0.001).astype(np.float16)

    @mb.program(input_specs=[mb.TensorSpec((T, I), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        wf = mb.constexpr_blockwise_shift_scale(data=w, scale=s, offset=None)
        return mb.matmul(x=x, y=wf, transpose_y=True, name="y")

    return prog


def build_v2x(T, O, I, ksplit=1, a8=False, seed=0):
    """V2 plus optional K-split (sum of ksplit chunks) and int8 activation quant/dequant."""
    rng = np.random.default_rng(seed)
    w = rng.integers(-127, 128, size=(O, I), dtype=np.int8)
    s = (rng.random((O, 1)) * 0.01 + 0.001).astype(np.float16)
    if I % ksplit:
        raise ValueError(f"inner dimension {I} is not divisible by ksplit {ksplit}")
    kc = I // ksplit

    @mb.program(input_specs=[mb.TensorSpec((T, I), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        if a8:
            xq = mb.quantize(input=x, scale=np.float16(0.02), output_dtype="int8")
            x = mb.dequantize(input=xq, scale=np.float16(0.02))
        wf = mb.constexpr_blockwise_shift_scale(data=w, scale=s, offset=None)
        if ksplit == 1:
            return mb.matmul(x=x, y=wf, transpose_y=True, name="y")
        acc = None
        for k in range(ksplit):
            xs = mb.slice_by_index(x=x, begin=[0, k * kc], end=[T, (k + 1) * kc])
            ws = mb.slice_by_index(x=wf, begin=[0, k * kc], end=[O, (k + 1) * kc])
            y = mb.matmul(x=xs, y=ws, transpose_y=True)
            acc = y if acc is None else mb.add(x=acc, y=y)
        return mb.identity(x=acc, name="y")

    return prog


def build_conv(T, O, I, ksplit=1, seed=0, a8=None, w_s=None, to=False):
    """1x1 Conv2d, channels-first (1, C, 1, T) layout, baked int8 weights, optional K-split."""
    rng = np.random.default_rng(seed)
    w = rng.integers(-127, 128, size=(O, I, 1, 1), dtype=np.int8)
    s = (rng.random((O, 1, 1, 1)) * 0.01 + 0.001).astype(np.float16)
    if w_s is not None:  # real weights: int8 [O, I] and fp16 per-channel scale [O]
        w = w_s[0].reshape(O, I, 1, 1).astype(np.int8)
        s = w_s[1].reshape(O, 1, 1, 1).astype(np.float16)
    if I % ksplit:
        raise ValueError(f"inner dimension {I} is not divisible by ksplit {ksplit}")
    kc = I // ksplit

    ws_np = [w[:, k * kc : (k + 1) * kc] for k in range(ksplit)]

    if a8 == "pre":

        @mb.program(
            input_specs=[mb.TensorSpec((1, I, 1, T), dtype=types.int8), mb.TensorSpec((1, 1, 1, T), dtype=types.fp16)],
            opset_version=TARGET,
        )
        def prog_pre(x, tok):
            acc = None
            for k in range(ksplit):
                xs = x if ksplit == 1 else mb.slice_by_index(x=x, begin=[0, k * kc, 0, 0], end=[1, (k + 1) * kc, 1, T])
                xs = mb.dequantize(input=xs, scale=np.float16(1.0))
                ws = mb.constexpr_blockwise_shift_scale(data=np.ascontiguousarray(ws_np[k]), scale=s, offset=None)
                y = mb.conv(x=xs, weight=ws, pad_type="valid")
                acc = y if acc is None else mb.add(x=acc, y=y)
            return mb.mul(x=acc, y=tok, name="y")

        return prog_pre

    if to:  # token-major (T, I) -> (T, O) interface; transposes happen inside the model

        @mb.program(input_specs=[mb.TensorSpec((T, I), dtype=types.fp16)], opset_version=TARGET)
        def prog_to(x):
            x = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, I, 1, T])
            acc = None
            for k in range(ksplit):
                xs = x if ksplit == 1 else mb.slice_by_index(x=x, begin=[0, k * kc, 0, 0], end=[1, (k + 1) * kc, 1, T])
                ws = mb.constexpr_blockwise_shift_scale(data=np.ascontiguousarray(ws_np[k]), scale=s, offset=None)
                y = mb.conv(x=xs, weight=ws, pad_type="valid")
                acc = y if acc is None else mb.add(x=acc, y=y)
            return mb.transpose(x=mb.reshape(x=acc, shape=[O, T]), perm=[1, 0], name="y")

        return prog_to

    @mb.program(input_specs=[mb.TensorSpec((1, I, 1, T), dtype=types.fp16)], opset_version=TARGET)
    def prog(x):
        if a8 == "static":
            xq = mb.quantize(input=x, scale=np.float16(0.02), output_dtype="int8")
            x = mb.dequantize(input=xq, scale=np.float16(0.02))
        elif a8 == "dyn":  # per-token (per column) scale, computed in-graph
            amax = mb.reduce_max(x=mb.abs(x=x), axes=[1], keep_dims=True)
            sc = mb.maximum(x=mb.real_div(x=amax, y=np.float16(127.0)), y=np.float16(1e-4))
            q = mb.clip(x=mb.round(x=mb.real_div(x=x, y=sc)), alpha=np.float16(-127), beta=np.float16(127))
            x = mb.mul(x=q, y=sc)
        acc = None
        for k in range(ksplit):
            xs = x if ksplit == 1 else mb.slice_by_index(x=x, begin=[0, k * kc, 0, 0], end=[1, (k + 1) * kc, 1, T])
            ws = mb.constexpr_blockwise_shift_scale(data=np.ascontiguousarray(ws_np[k]), scale=s, offset=None)
            y = mb.conv(x=xs, weight=ws, pad_type="valid")
            acc = y if acc is None else mb.add(x=acc, y=y)
        return mb.identity(x=acc, name="y")

    return prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("tokens", type=int)
    ap.add_argument("out", type=int)
    ap.add_argument("inn", type=int)
    ap.add_argument(
        "--variant",
        default="v1",
        choices=[
            "v1",
            "v2",
            "v2k2",
            "v2k4",
            "v2a8",
            "v2a8k2",
            "c1",
            "c2",
            "c4",
            "c1s",
            "c4s",
            "c1d",
            "c4d",
            "c1p",
            "c4p",
            "t1",
            "t4",
        ],
    )
    a = ap.parse_args()
    if a.variant == "v1":
        prog = build_v1(a.tokens, a.out, a.inn)
    elif a.variant == "v2":
        prog = build_v2(a.tokens, a.out, a.inn)
    elif a.variant[0] in "ct":
        prog = build_conv(
            a.tokens,
            a.out,
            a.inn,
            ksplit=int(a.variant[1]),
            a8={"s": "static", "d": "dyn", "p": "pre"}.get(a.variant[2:]),
            to=a.variant[0] == "t",
        )
    else:
        ks = int(a.variant[-1]) if a.variant[-2] == "k" else 1
        prog = build_v2x(a.tokens, a.out, a.inn, ksplit=ks, a8=a.variant.startswith("v2a8"))
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    os.makedirs(a.outdir, exist_ok=True)
    name = f"{a.variant}_{a.tokens}x{a.out}x{a.inn}.mlpackage"
    m.save(os.path.join(a.outdir, name))
    print(name)


if __name__ == "__main__":
    main()
