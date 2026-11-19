import numpy as np
import pytest

from mlp_program import (
    MlpSpec, epilogue_gain, plan_split, program_scales, quantize_per_row, reference_fp16, reference_fp32,
)

D, H, M = 384, 1024, 256  # scaled-down dims, same structure


def _case(seed=0):
    r = np.random.default_rng(seed)
    x = r.standard_normal((1, M, D)).astype(np.float32)
    w1, w3 = (r.standard_normal((H, D)).astype(np.float32) * 0.02 for _ in range(2))
    w2 = r.standard_normal((D, H)).astype(np.float32) * 0.02
    return x, w1, w3, w2


def _cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("scale_mode,w2_scale", [("weight", "program"), ("output", "program"), ("output", "epilogue")])
def test_mirror_matches_fp32(scale_mode, w2_scale):
    spec = MlpSpec(mpad=M, dim=D, hidden=H, scale_mode=scale_mode, w2_scale=w2_scale, k13_chunks=1, k2_chunks=4)
    x, w1, w3, w2 = _case()
    (q1, s1), (q3, s3), (q2, s2) = quantize_per_row(w1), quantize_per_row(w3), quantize_per_row(w2)
    y, stats = reference_fp16(x, {"w1": q1, "w3": q3, "w2": q2}, program_scales(spec, s1, s3, s2), spec)
    out = y.astype(np.float32) * epilogue_gain(spec, s2)
    assert stats["nonfinite"] == 0
    assert _cos(out, reference_fp32(x, w1, w3, w2)) > 0.999


def test_headroom_prevents_overflow():
    spec = MlpSpec(mpad=M, dim=D, hidden=H, k2_chunks=4)
    r = np.random.default_rng(1)
    x = r.standard_normal((1, M, D)).astype(np.float32) * 8
    w1, w3 = (r.standard_normal((H, D)).astype(np.float32) * 0.3 for _ in range(2))
    w2 = r.standard_normal((D, H)).astype(np.float32) * 0.3
    (q1, s1), (q3, s3), (q2, s2) = quantize_per_row(w1), quantize_per_row(w3), quantize_per_row(w2)
    _, stats = reference_fp16(x, {"w1": q1, "w3": q3, "w2": q2}, program_scales(spec, s1, s3, s2), spec)
    assert stats["nonfinite"] == 0 and stats["max_h"] < 65504


def test_plan_split():
    assert plan_split(4224, 0.25) == (1024, 3200)
    assert plan_split(4224, 0.30) == (1280, 2944)
    assert plan_split(4224, 0.15)[0] % 256 == 0


def test_mil_builds():
    pytest.importorskip("coremltools")
    from mlp_program import build_mil_program

    for spec in (
        MlpSpec(mpad=M, dim=D, hidden=H, k2_chunks=4),
        MlpSpec(mpad=M, dim=D, hidden=H, scale_mode="output", w2_scale="epilogue", k13_chunks=2),
        MlpSpec(mpad=M, dim=D, hidden=H, weight_dtype="fp16"),
    ):
        build_mil_program(spec)
