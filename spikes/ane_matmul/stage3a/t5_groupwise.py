"""Test 5: group-wise int8 on the ANE that reproduces MLX q8 (group 64, affine).

Steps:
1. Discover the exact MLX affine dequant formula for bits=8.
2. Map (code, scale, bias) onto Core ML constexpr_blockwise_shift_scale:
   dequant = scale * (data - offset), group 64 along the inner dimension.
3. Weight-level error of the mapping, in fp64.
4. Output-level comparison: ANE conv against mx.quantized_matmul, same codes.
5. Speed and ANE placement against per-channel int8 (the Stage 1 form).
Shape: o layer 3840x3840 at T=4224. Synthetic weights.
"""

import os
import tempfile

import gen3a
import numpy as np
import s3a
from s3a import ANERUN, MODELS, WATCHDOG, BenchLock, anebench, aneplan, emit, gen_fixed, run

T = s3a.T_FIXED
O, I = 3840, 3840
GROUP = 64


def unpack_codes(wq, rows, cols):
    """MLX bits=8 packs 4 codes per uint32 word; return the [rows, cols] uint8 codes."""
    a = np.array(wq)
    if a.dtype == np.uint8:
        return a.reshape(rows, cols)
    return a.view(np.uint8).reshape(rows, cols)  # little-endian: byte j holds element j


def discover_mlx_format():
    """Return (formula_name, transform) that reproduces mx.dequantize bitwise."""
    import mlx.core as mx

    rng = np.random.default_rng(3)
    w = mx.array((rng.normal(size=(64, 3840)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=GROUP, bits=8)
    deq = np.array(mx.dequantize(wq, s, b, group_size=GROUP, bits=8).astype(mx.float32)).astype(np.float64)
    c = unpack_codes(wq, 64, 3840).astype(np.float64)
    s64 = np.array(s.astype(mx.float32)).astype(np.float64)
    b64 = np.array(b.astype(mx.float32)).astype(np.float64)
    sg = np.repeat(s64, GROUP, axis=1)
    bg = np.repeat(b64, GROUP, axis=1)
    formulas = {
        "s*c+b": sg * c + bg,
        "s*(c-128)+b": sg * (c - 128.0) + bg,
        "s*int8(c)+b": sg * np.where(c > 127, c - 256, c) + bg,
    }
    best = None
    for name, out in formulas.items():
        # mx.dequantize returns bf16: round the fp64 candidate to bf16 before comparing.
        out_bf16 = np.array(mx.array(out).astype(mx.bfloat16).astype(mx.float32)).astype(np.float64)
        err = float(np.abs(out_bf16 - deq).max())
        if best is None or err < best[1]:
            best = (name, err)
    return w, wq, s, b, deq, best, c, s64, b64


def anerun_predict(model, x_np):
    """Run one channels-first W8A16 conv model on x_np [T, I] fp32; return y [T, O] fp32."""
    with tempfile.TemporaryDirectory() as td:
        x_cf = np.ascontiguousarray(x_np.astype(np.float16).T)  # [I, T] == [1, I, 1, T] row-major
        x_path = os.path.join(td, "x.bin")
        y_path = os.path.join(td, "y.bin")
        x_cf.tofile(x_path)
        run([WATCHDOG, ANERUN, model, y_path, x_path], timeout=600)
        y = np.fromfile(y_path, dtype=np.float16).reshape(O, T)
        return np.ascontiguousarray(y.T).astype(np.float64)


def build_bw(name, q4, scale, offset):
    p = MODELS / f"{name}_{T}x{O}x{I}.mlpackage"
    if not p.exists():
        prog = gen3a.build_blockwise(T, O, I, q4, scale, offset)
        gen3a.convert(prog, p)
    return p


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx

    w, wq, s, b, deq, best, c, s64, b64 = discover_mlx_format()
    emit(
        "t5_groupwise",
        dict(
            phase="format",
            best_formula=best[0],
            max_abs_err_vs_dequantize=best[1],
            wq_dtype=str(wq.dtype),
            wq_shape=list(wq.shape),
            scale_shape=list(s.shape),
            scale_dtype=str(s.dtype),
            bias_dtype=str(b.dtype),
        ),
    )
    if best[1] > 1e-3:  # one bf16 ulp at |w|~0.08 is ~3e-4; allow two
        emit(
            "t5_groupwise",
            dict(phase="format", note="no candidate formula reproduces mx.dequantize; aborting numerics"),
        )
        return

    # Full-size weights for the model.
    rng = np.random.default_rng(11)
    w_full = mx.array((rng.normal(size=(O, I)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq_f, s_f, b_f = mx.quantize(w_full, group_size=GROUP, bits=8)
    c_f = unpack_codes(wq_f, O, I)
    s_f64 = np.array(s_f.astype(mx.float32)).astype(np.float64)
    b_f64 = np.array(b_f.astype(mx.float32)).astype(np.float64)

    # Map to Core ML: dequant = scale * (data - offset). For "s*c+b": offset = -b/s (uint8 data).
    # For "s*(c-128)+b": offset = 128 - b/s. Try the discovered formula.
    data = c_f.astype(np.int8) if best[0] == "s*int8(c)+b" else c_f  # keep uint8 codes otherwise
    base = 128.0 if best[0] == "s*(c-128)+b" else 0.0
    off_f64 = base - b_f64 / s_f64  # [O, I/64]
    scale4 = np.ascontiguousarray(s_f64.astype(np.float16)).reshape(O, I // GROUP, 1, 1)
    off4_16 = np.ascontiguousarray(off_f64.astype(np.float16)).reshape(O, I // GROUP, 1, 1)
    off4_32 = np.ascontiguousarray(off_f64.astype(np.float32)).reshape(O, I // GROUP, 1, 1)
    data4 = np.ascontiguousarray(data).reshape(O, I, 1, 1)

    # Weight-level representation error in fp64: scale * (data - offset) against the MLX dequant.
    code = data.astype(np.float64)
    if best[0] == "s*(c-128)+b":
        ref_w = np.repeat(s_f64, GROUP, axis=1) * (c_f.astype(np.float64) - 128.0) + np.repeat(b_f64, GROUP, axis=1)
    elif best[0] == "s*int8(c)+b":
        ref_w = np.repeat(s_f64, GROUP, axis=1) * code + np.repeat(b_f64, GROUP, axis=1)
    else:
        ref_w = np.repeat(s_f64, GROUP, axis=1) * c_f.astype(np.float64) + np.repeat(b_f64, GROUP, axis=1)
    an_w_16 = np.repeat(scale4.reshape(O, I // GROUP).astype(np.float64), GROUP, axis=1) * (
        code - np.repeat(off4_16.reshape(O, I // GROUP).astype(np.float64), GROUP, axis=1)
    )
    an_w_32 = np.repeat(scale4.reshape(O, I // GROUP).astype(np.float64), GROUP, axis=1) * (
        code - np.repeat(off4_32.reshape(O, I // GROUP).astype(np.float64), GROUP, axis=1)
    )
    rel = float(np.abs(ref_w).max())
    emit(
        "t5_groupwise",
        dict(
            phase="weights",
            max_abs_err_offset16=float(np.abs(an_w_16 - ref_w).max()),
            max_abs_err_offset32=float(np.abs(an_w_32 - ref_w).max()),
            max_abs_weight=float(rel),
            rel_err_offset16=float(np.abs(an_w_16 - ref_w).max() / rel),
        ),
    )

    # Output-level: ANE conv against an fp64 emulation of the MLX q8 dequant on the
    # same fp16 input, plus mx.quantized_matmul (bf16 x) as the production reference.
    x_np = rng.normal(size=(T, I)).astype(np.float16)
    deq_w = ref_w.T  # [I, O] fp64, the exact MLX-dequantised weights
    emul = x_np.astype(np.float64) @ deq_w
    ref = np.array(
        mx.quantized_matmul(
            mx.array(x_np).astype(mx.bfloat16), wq_f, s_f, b_f, transpose=True, group_size=GROUP, bits=8
        ).astype(mx.float32)
    ).astype(np.float64)

    p16 = build_bw("bw16", data4, scale4, off4_16)
    y16 = anerun_predict(p16, x_np)

    def cosine(a, b):
        return float(a.reshape(-1) @ b.reshape(-1) / (np.linalg.norm(a) * np.linalg.norm(b)))

    emit(
        "t5_groupwise",
        dict(
            phase="outputs",
            ane_vs_emulation_cosine=cosine(y16, emul),
            ane_vs_emulation_maxrel=float(np.abs(y16 - emul).max() / np.abs(emul).max()),
            mlxq8_vs_emulation_cosine=cosine(ref, emul),
            ane_vs_mlxq8_cosine=cosine(y16, ref),
            n_inf=int(np.isinf(y16).sum()),
            n_nan=int(np.isnan(y16).sum()),
        ),
    )

    emit(
        "t5_groupwise", dict(phase="plan", blockwise=aneplan(p16), perchannel=aneplan(gen_fixed(MODELS, T, O, I, "c1")))
    )

    with BenchLock(owner=f"stage3a-t5-{os.getpid()}"):
        bw = anebench(p16, shape=(T, O, I), iters=50)
        pc = anebench(MODELS / f"c1_{T}x{O}x{I}.mlpackage", shape=(T, O, I), iters=50)
        emit(
            "t5_groupwise",
            dict(
                phase="speed",
                blockwise_ms=bw["median_ms"],
                perchannel_ms=pc["median_ms"],
                blockwise_tflops=bw["tflops"],
                perchannel_tflops=pc["tflops"],
                ratio=pc["median_ms"] / bw["median_ms"],
            ),
        )


if __name__ == "__main__":
    main()
