"""Test 7: W8A8 overflow safety.

A. Graph structure: op order and device per W8A8 variant (static, dynamic, pre-quantised).
B. Accumulation probe: unit-scale weights, adversarial +/-127 int8 activations whose
   exact sum is 0 but whose partial sums exceed the fp16 maximum. An int32 accumulator
   returns 0; an fp16 accumulator saturates.
C. Outlier tokens with realistic scales: dynamic per-token variant against an fp64
   emulation of the same quantisation; checks for inf/nan and measures the error.
All correctness checks; no timed runs, no lock.
"""

import os
import sys
import tempfile

import numpy as np
import s3a

sys.path.insert(0, str(s3a.SPIKE))
import gen_models as G  # noqa: E402
from s3a import MODELS, aneplan, emit, run

T_SMALL = 64
O_BIG, I = 3840, 3840


def _convert(prog, path):
    import coremltools as ct

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=G.TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    m.save(str(path))
    return path


def mil_ops(prog):
    fn = prog.functions["main"] if "main" in prog.functions else list(prog.functions.values())[0]
    return [op.op_type for op in fn.operations]


def run_model(model, x_file_bytes, tok_bytes=None, O=O_BIG, T=T_SMALL):
    with tempfile.TemporaryDirectory() as td:
        x_path = os.path.join(td, "x.bin")
        y_path = os.path.join(td, "y.bin")
        with open(x_path, "wb") as f:
            f.write(x_file_bytes)
        cmd = [s3a.ANERUN, str(model), y_path, x_path]
        if tok_bytes is not None:
            tok = os.path.join(td, "tok.bin")
            with open(tok, "wb") as f:
                f.write(tok_bytes)
            cmd.append(tok)
        run(cmd, timeout=600)
        return np.fromfile(y_path, dtype=np.float16).reshape(O, T).T.astype(np.float64)


def phase_a():
    for variant in ("c1s", "c1d", "c1p"):
        prog = G.build_conv(
            T_SMALL, O_BIG, I, ksplit=1, a8={"c1s": "static", "c1d": "dyn", "c1p": "pre"}.get(variant.replace("c1", ""))
        )
        emit("t7_w8a8", dict(phase="graph", variant=variant, mil_ops=mil_ops(prog)))
        path = MODELS / f"{variant}_{T_SMALL}x{O_BIG}x{I}.mlpackage"
        if not path.exists():
            _convert(prog, path)
        emit("t7_w8a8", dict(phase="graph", variant=variant, plan=aneplan(path)))


def phase_b():
    """Unit-scale weights, adversarial activations: exact answer is 0, fp16 partials overflow."""
    ones_w = np.ones((O_BIG, I), dtype=np.int8)
    ones_s = np.ones((O_BIG, 1), dtype=np.float16)
    # Pattern: +127 for the first half of I, -127 for the second half.
    pat = np.concatenate([np.full(I // 2, 127, dtype=np.int8), np.full(I // 2, -127, dtype=np.int8)])
    variants = {}
    # pre-quantised int8 input variant (the production W8A8 form)
    prog = G.build_conv(T_SMALL, O_BIG, I, ksplit=1, a8="pre", w_s=(ones_w, ones_s))
    p_pre = MODELS / f"unit_pre_{T_SMALL}x{O_BIG}x{I}.mlpackage"
    if not p_pre.exists():
        _convert(prog, p_pre)
    x_i8 = np.tile(pat, (T_SMALL, 1))  # [T, I] int8
    tok = np.ones((1, 1, 1, T_SMALL), dtype=np.float16)
    y = run_model(p_pre, x_i8.astype(np.int8).T.tobytes(), tok.tobytes())
    variants["pre_int8_unit_scale"] = y
    # W8A16 variant: fp16 activations, unit int8 weights
    prog = G.build_conv(T_SMALL, O_BIG, I, ksplit=1, w_s=(ones_w, ones_s))
    p_16 = MODELS / f"unit_a16_{T_SMALL}x{O_BIG}x{I}.mlpackage"
    if not p_16.exists():
        _convert(prog, p_16)
    x_f16 = np.tile(pat.astype(np.float16), (T_SMALL, 1))
    y = run_model(p_16, np.ascontiguousarray(x_f16.T).tobytes())
    variants["w8a16_unit_scale"] = y
    for name, y in variants.items():
        emit(
            "t7_w8a8",
            dict(
                phase="accumulation",
                case=name,
                max_abs_y=float(np.abs(y).max()),
                n_inf=int(np.isinf(y).sum()),
                n_nan=int(np.isnan(y).sum()),
                exact_expected=0.0,
                verdict="int32-like accumulation"
                if np.all(np.abs(y[np.isfinite(y)]) < 1.0) and not np.isinf(y).any()
                else "fp16 saturation or residual",
            ),
        )


def phase_c():
    """Realistic scales with an outlier token, dynamic per-token W8A8, against fp64 emulation."""
    rng = np.random.default_rng(9)
    w = (rng.normal(size=(O_BIG, I)) * 0.02).astype(np.float64)
    sc_w = np.abs(w).max(axis=1) / 127.0
    wq = np.clip(np.round(w / sc_w[:, None]), -127, 127)
    sc_w16 = sc_w.astype(np.float16)
    x = rng.normal(size=(T_SMALL, I))
    x[3] *= 300.0  # outlier token
    x16 = x.astype(np.float16).astype(np.float64)

    prog = G.build_conv(T_SMALL, O_BIG, I, ksplit=1, a8="dyn", w_s=(wq.astype(np.int8), sc_w16))
    emit("t7_w8a8", dict(phase="outlier", mil_ops=mil_ops(prog)))
    path = MODELS / f"outlier_c1d_{T_SMALL}x{O_BIG}x{I}.mlpackage"
    if not path.exists():
        _convert(prog, path)
    y = run_model(path, np.ascontiguousarray(x16.astype(np.float16).T).tobytes())

    # fp64 emulation of the in-graph dynamic quantisation, then the int8 matmul in fp64.
    amax = np.abs(x16).max(axis=1, keepdims=True)
    sc_t = np.maximum(amax / 127.0, 1e-4)
    q = np.clip(np.round(x16 / sc_t), -127, 127)
    xd = q * sc_t
    w_deq = wq * sc_w[:, None].astype(np.float64)
    ref = xd @ w_deq.T
    cos = float(y.reshape(-1) @ ref.reshape(-1) / (np.linalg.norm(y) * np.linalg.norm(ref)))
    emit(
        "t7_w8a8",
        dict(
            phase="outlier",
            cosine_vs_emulation=cos,
            max_abs_diff=float(np.abs(y - ref).max()),
            max_abs_ref=float(np.abs(ref).max()),
            n_inf=int(np.isinf(y).sum()),
            n_nan=int(np.isnan(y).sum()),
            outlier_token_max_abs_y=float(np.abs(y[3]).max()),
        ),
    )


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    phase_a()
    phase_b()
    phase_c()


if __name__ == "__main__":
    main()
