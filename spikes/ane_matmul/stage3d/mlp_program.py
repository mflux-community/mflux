"""Stage 3d: fused SwiGLU MLP as one Core ML (MIL) program, weights as inputs.

Layout: matmul form, x is [1, Mpad, D] fp16 (rank 3), weights are [out, in] (nn.Linear layout).
Math (mirrors FeedForward in z_image):  y/32 = ((silu(x.w1^T) * (x.w3^T / 32)) . w2^T)
The /32 headroom on the w3 branch keeps the SwiGLU product and the w2 sums inside fp16.
The program returns y/32; the GPU epilogue multiplies by 32 (and by w2 scales when w2_scale="epilogue") in fp32.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 3840
HIDDEN = 10240
HEADROOM = 32.0
QUANTUM = 256


@dataclass(frozen=True)
class MlpSpec:
    mpad: int
    dim: int = DIM
    hidden: int = HIDDEN
    weight_dtype: str = "int8"  # "int8" | "fp16" (fallback; int8 model inputs need macOS 26 / iOS26 target)
    # "weight": dequantise rows in-program (w * s) before the matmul; safe range, extra elementwise work.
    # "output": matmul on raw int8 values with x pre-shrunk by 1/INPUT_SHRINK, scale applied to the matmul output.
    scale_mode: str = "weight"
    w2_scale: str = "program"  # "program" | "epilogue" (epilogue only valid with scale_mode="output")
    k13_chunks: int = 1  # K=3840 for w1/w3; set 2 if the ANE rejects/slow-paths it
    k2_chunks: int = 4  # K=10240 for w2
    # The a+b range fix from probe item 1 (report section 5h), needed for real weights:
    # (a) dequantise weights by a constant 1/sqrt(K) so a matmul's partial sums stop scaling with
    #     K and the weight magnitude, and carry the real per-row scale on the output instead;
    # (b) divide h by its own per-row max before the w2 matmul and restore that factor in the fp32
    #     GPU epilogue. Restoring it inside the program overflows fp16, so h_max is a 2nd output.
    # Without it, real trained weights drive partial sums to 3.6e7 against fp16's 65504 ceiling.
    range_fix: bool = False

    @property
    def name(self) -> str:
        return (
            f"mlp_m{self.mpad}_{self.weight_dtype}_{self.scale_mode}_w2{self.w2_scale}"
            f"_k{self.k13_chunks}x{self.k2_chunks}{'_rf' if self.range_fix else ''}"
        )

    def validate(self) -> None:
        assert self.mpad % QUANTUM == 0, "Mpad must be a multiple of 256"
        assert self.scale_mode in ("weight", "output")
        assert self.w2_scale in ("program", "epilogue")
        assert not (self.w2_scale == "epilogue" and self.scale_mode == "weight"), "epilogue w2 scale needs output mode"
        assert self.dim % self.k13_chunks == 0 and self.hidden % self.k2_chunks == 0
        # The fix is defined against the int8 code range (+-127), so it needs int8 weights.
        assert not self.range_fix or self.weight_dtype == "int8", "range_fix needs int8 weight inputs"
        assert not (self.range_fix and self.w2_scale == "epilogue"), "range_fix keeps the w2 scale in-program"


# Raw int8 products reach ~127*|x|*sqrt(K); shrink x so fp16 partial sums stay far from 65504.
INPUT_SHRINK = 128.0


def plan_split(seq_len: int, share: float, quantum: int = QUANTUM) -> tuple[int, int]:
    """Return (ane_rows, gpu_rows). ANE rows are the nearest multiple of 256 to share*S, at least 256."""
    ane_rows = max(quantum, int(round(seq_len * share / quantum)) * quantum)
    ane_rows = min(ane_rows, (seq_len // quantum) * quantum)
    return ane_rows, seq_len - ane_rows


def quantize_per_row(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel int8. w: [out, in] float -> (int8 [out, in], scale fp32 [out])."""
    w = w.astype(np.float32)
    scale = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def program_scales(spec: MlpSpec, s1: np.ndarray, s3: np.ndarray, s2: np.ndarray) -> dict[str, np.ndarray]:
    """Host-side scale vectors fed to the program (fp16). Output mode folds the x pre-shrink into
    the s1/s3 scales only, because x is the sole input that gets shrunk.

    s2 must NOT carry the fold: its matmul consumes h (the SwiGLU product), which is already
    restored by s1/s3. Folding it too made the program output 128x too large, and the numpy
    mirror inherited the same reasoning, so mirror-vs-program agreed and both disagreed with the
    real FFN -- and T1's cosine, being scale-invariant, scored it 0.9998 and passed it.
    """
    fold = INPUT_SHRINK if spec.scale_mode == "output" else 1.0
    if spec.range_fix:
        # (a) weights are dequantised by a constant 1/sqrt(K) in-program, so the real per-row scale
        # has to arrive on the output side: s * sqrt(K) restores exactly what the constant removed.
        sq1, sq2 = float(np.sqrt(spec.dim)), float(np.sqrt(spec.hidden))
        return {
            "s1": (s1 * sq1).astype(np.float16),
            "s3": (s3 * sq1).astype(np.float16),
            "s2": (s2 * sq2).astype(np.float16),
        }
    return {
        "s1": (s1 * fold).astype(np.float16),
        "s3": (s3 * fold).astype(np.float16),
        "s2": s2.astype(np.float16),
    }


def epilogue_gain(spec: MlpSpec, s2: np.ndarray) -> np.ndarray:
    """fp32 per-channel gain applied by the GPU to the program output [.., D]."""
    gain = np.full(spec.dim, HEADROOM, dtype=np.float32)
    if spec.w2_scale == "epilogue":
        gain = gain * s2.astype(np.float32)  # no INPUT_SHRINK: see program_scales
    return gain


def build_mil_program(spec: MlpSpec):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    spec.validate()
    M, D, H = spec.mpad, spec.dim, spec.hidden
    wdt = types.int8 if spec.weight_dtype == "int8" else types.fp16
    specs = {
        "x_rows": mb.TensorSpec(shape=(1, M, D), dtype=types.fp16),
        "w1": mb.TensorSpec(shape=(H, D), dtype=wdt),
        "w3": mb.TensorSpec(shape=(H, D), dtype=wdt),
        "w2": mb.TensorSpec(shape=(D, H), dtype=wdt),
        "s1": mb.TensorSpec(shape=(H,), dtype=types.fp16),
        "s3": mb.TensorSpec(shape=(H,), dtype=types.fp16),
        "s2": mb.TensorSpec(shape=(D,), dtype=types.fp16),
    }

    def linear(x, w, scale, k_chunks, in_dim, out_dim, scale_in_program, prefix):
        # (a) The range fix widens int8 with 1/sqrt(K) instead of 1.0, so a raw code dot product
        # stays near |x| rather than |x|*127*sqrt(K). cast() rejects int8, so dequantize with a
        # constant scale is the int8 -> fp16 widening (needs the iOS26 target).
        dq = np.float16(1.0 / np.sqrt(in_dim)) if spec.range_fix else np.float16(1.0)
        wf = mb.dequantize(input=w, scale=dq, name=f"{prefix}_wf") if spec.weight_dtype == "int8" else w
        if spec.scale_mode == "weight" and scale_in_program:
            wf = mb.mul(x=wf, y=mb.reshape(x=scale, shape=(out_dim, 1)), name=f"{prefix}_wdeq")
        step = in_dim // k_chunks
        acc = None
        for i in range(k_chunks):
            lo, hi = i * step, (i + 1) * step
            xi = x if k_chunks == 1 else mb.slice_by_index(x=x, begin=[0, 0, lo], end=[1, M, hi], name=f"{prefix}_x{i}")
            wi = (
                wf
                if k_chunks == 1
                else mb.slice_by_index(x=wf, begin=[0, lo], end=[out_dim, hi], name=f"{prefix}_w{i}")
            )
            yi = mb.matmul(x=xi, y=wi, transpose_y=True, name=f"{prefix}_mm{i}")
            acc = yi if acc is None else mb.add(x=acc, y=yi, name=f"{prefix}_acc{i}")
        # With the range fix the real scale always lands on the output: it cannot go on the weights,
        # because the constant 1/sqrt(K) is what keeps the partial sums small.
        if (spec.scale_mode == "output" or spec.range_fix) and scale_in_program:
            acc = mb.mul(x=acc, y=scale, name=f"{prefix}_oscale")
        return acc

    opset = ct.target.iOS26 if spec.weight_dtype == "int8" else ct.target.iOS18

    @mb.program(input_specs=list(specs.values()), opset_version=opset)
    def prog(x_rows, w1, w3, w2, s1, s3, s2):
        x = x_rows
        if spec.scale_mode == "output" and not spec.range_fix:
            x = mb.mul(x=x, y=np.float16(1.0 / INPUT_SHRINK), name="x_shrink")
        a = linear(x, w1, s1, spec.k13_chunks, D, H, True, "w1")
        b = linear(x, w3, s3, spec.k13_chunks, D, H, True, "w3")
        b = mb.mul(x=b, y=np.float16(1.0 / HEADROOM), name="b_headroom")
        h = mb.mul(x=mb.silu(x=a, name="w1_silu"), y=b, name="swiglu")
        hmax = None
        if spec.range_fix:
            # (b) Normalise h by its own per-row max. The floor is the smallest fp16 normal, not
            # 1e-8, which underflows to zero in fp16 and would leave a zero row as 0/0.
            hmax = mb.reduce_max(x=mb.abs(x=h, name="h_abs"), axes=[-1], keep_dims=True, name="h_absmax")
            hmax = mb.maximum(x=hmax, y=np.float16(6.1e-5), name="h_max")
            h = mb.real_div(x=h, y=hmax, name="h_norm")
        y = linear(h, w2, s2, spec.k2_chunks, H, D, spec.w2_scale == "program", "w2")
        y = mb.identity(x=y, name="y_rows")
        if spec.range_fix:
            # h_max leaves as a second output on purpose: restoring it inside the program is what
            # overflows fp16 (see the module docstring), so the fp32 GPU epilogue does it.
            return y, mb.identity(x=hmax, name="h_max_out")
        return y

    return prog, list(specs)


def convert_to_mlpackage(spec: MlpSpec, out_dir: str):
    import coremltools as ct

    prog, names = build_mil_program(spec)
    target = ct.target.iOS26 if spec.weight_dtype == "int8" else ct.target.iOS18  # iOS26 == macOS 26
    model = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=target,
    )
    path = f"{out_dir}/{spec.name}.mlpackage"
    model.save(path)
    return path


def reference_fp16(x: np.ndarray, q: dict, scales: dict, spec: MlpSpec) -> tuple[np.ndarray, dict]:
    """Numpy mirror of the program with fp16 rounding at every op boundary. Returns (y/32 fp16, stats).

    q: {"w1","w3","w2"} int8 (or fp16 when weight_dtype="fp16"); scales: output of program_scales().
    """
    f16 = np.float16
    stats: dict = {"max_part": {}}

    def mm(a, w, chunks, tag=""):
        step = a.shape[-1] // chunks
        acc = None
        for i in range(chunks):
            part32 = (
                a[..., i * step : (i + 1) * step].astype(np.float32)
                @ w[:, i * step : (i + 1) * step].astype(np.float32).T
            )
            # The untruncated partial sum is the fp16 range risk, so record it before the cast:
            # the ANE clamps at the ceiling instead of producing inf, so finiteness proves nothing.
            p = float(np.abs(part32).max())
            stats["max_part"][tag] = max(stats["max_part"].get(tag, 0.0), p)
            part = part32.astype(f16)
            acc = part if acc is None else (acc + part).astype(f16)
        return acc

    def lin(a, w, s, chunks, scale_in_program, tag=""):
        wf = w.astype(f16)
        if spec.range_fix:
            # (a) the same constant the program dequantises by; the real scale arrives on the output
            wf = (wf * f16(1.0 / np.sqrt(a.shape[-1]))).astype(f16)
        elif spec.scale_mode == "weight" and scale_in_program:
            wf = (wf * s.astype(f16)[:, None]).astype(f16)
        out = mm(a, wf, chunks, tag)
        if (spec.scale_mode == "output" or spec.range_fix) and scale_in_program:
            out = (out * s.astype(f16)).astype(f16)
        return out

    xin = x.astype(f16)
    if spec.scale_mode == "output" and not spec.range_fix:
        xin = (xin * f16(1.0 / INPUT_SHRINK)).astype(f16)
    a = lin(xin, q["w1"], scales["s1"], spec.k13_chunks, True, "w1")
    b = (lin(xin, q["w3"], scales["s3"], spec.k13_chunks, True, "w3") * f16(1.0 / HEADROOM)).astype(f16)
    silu = (a.astype(np.float32) / (1.0 + np.exp(-np.clip(a.astype(np.float32), -80, 80)))).astype(f16)
    h = (silu * b).astype(f16)
    rowmax = None
    if spec.range_fix:
        # (b) per-row normalisation; 6.1e-5 is fp16's smallest normal, so a zero row cannot give 0/0
        rowmax = np.maximum(np.abs(h.astype(np.float32)).max(axis=-1, keepdims=True), float(f16(6.1e-5))).astype(f16)
        h = (h / rowmax).astype(f16)
    y = lin(h, q["w2"], scales["s2"], spec.k2_chunks, spec.w2_scale == "program", "w2")
    stats.update(
        max_a=float(np.abs(a).max()),
        max_h=float(np.abs(h).max()),
        max_y=float(np.abs(y).max()),
        nonfinite=int((~np.isfinite(y)).sum()),
        max_part_over_all=max(stats["max_part"].values() or [0.0]),
        # the caller's fp32 epilogue must multiply y by this, exactly as the bridge returns h_max_out
        rowmax=rowmax,
    )
    return y, stats


def reference_fp32(x: np.ndarray, w1: np.ndarray, w3: np.ndarray, w2: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    a = x @ w1.astype(np.float32).T
    b = x @ w3.astype(np.float32).T
    h = (a / (1.0 + np.exp(-a))) * b
    return h @ w2.astype(np.float32).T
