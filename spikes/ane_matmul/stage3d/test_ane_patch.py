"""Weight-free smoke test for the --ane-mlp wiring.

No trained weights and no ANE: the mirror backend is numpy emulation of the same program, so these
assertions cover the wiring (what gets replaced, what the epilogue restores, that unpatching puts
the ORIGINAL module back) rather than the arithmetic, which test_mlp_program.py and the report cover.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SPIKE = Path(__file__).resolve().parent
sys.path.insert(0, str(SPIKE))

mx = pytest.importorskip("mlx.core")

import ane_patch  # noqa: E402
from rowsplit import CallStats, RowSplitFFN  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward  # noqa: E402

DIM, HID, SEQ, MPAD = 384, 1024, 512, 256


class TinyTransformer:
    def __init__(self, n=3):
        self.layers = [type("L", (), {"feed_forward": FeedForward(DIM, HID)})() for _ in range(n)]
        self.noise_refiner = []
        self.context_refiner = []

    def parameters(self):
        # the real transformer returns [(path, mx.array)]; unpatch evaluates it
        return []


class TinyModel:
    def __init__(self):
        self.transformer = TinyTransformer()


def _h():
    r = np.random.default_rng(3)
    return mx.array(r.standard_normal((1, SEQ, DIM)).astype(np.float32)).astype(mx.bfloat16)


def _patch(model, range_fix=True):
    spec = ane_patch.spec_for(MPAD / SEQ, SEQ, range_fix=range_fix, dim=DIM, hidden=HID)
    assert spec.mpad == MPAD
    return ane_patch._patch_transformer(model.transformer, ane_patch.make_backend("mirror"), spec, CallStats()), spec


def test_patch_replaces_every_main_block_and_keeps_the_original():
    model = TinyModel()
    originals = [l.feed_forward for l in model.transformer.layers]
    n, _ = _patch(model)
    assert n == len(model.transformer.layers) == 3
    for layer, orig in zip(model.transformer.layers, originals, strict=True):
        assert isinstance(layer.feed_forward, RowSplitFFN)
        # unpatching must be able to hand back the SAME module, never a fresh one
        assert layer.feed_forward.ff is orig


def test_patched_ffn_matches_the_plain_ffn_within_int8_tolerance():
    model = TinyModel()
    _, spec = _patch(model)
    h = _h()
    orig = model.transformer.layers[0].feed_forward.ff
    y_ref = np.array(orig(h).astype(mx.float32))
    y_patch = np.array(model.transformer.layers[0].feed_forward(h).astype(mx.float32))
    assert y_patch.shape == y_ref.shape
    a, b = y_patch.astype(np.float64).ravel(), y_ref.astype(np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    relerr = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    # cosine alone would pass a constant-gain error, so both are checked (see report section 5c)
    assert cos > 0.999, cos
    assert relerr < 0.05, relerr


def test_stats_record_the_real_token_count_not_just_the_intended_one():
    """An image time is meaningless unless the S the split actually saw is recorded."""
    model = TinyModel()
    stats = CallStats()
    spec = ane_patch.spec_for(MPAD / SEQ, SEQ, dim=DIM, hidden=HID)
    ane_patch._patch_transformer(model.transformer, ane_patch.make_backend("mirror"), spec, stats)
    model.transformer.layers[0].feed_forward(_h())
    assert stats.seq_seen() == [SEQ], stats.seq_seen()
    # summary_ms must ignore the seq set rather than try to take a median of it
    summary = stats.summary_ms()
    assert set(summary) == {"prep", "wait", "epilogue"}, summary
    assert all(v > 0 for v in summary.values()), summary


def test_unpatch_restores_the_original_modules_not_fresh_ones():
    model = TinyModel()
    originals = [l.feed_forward for l in model.transformer.layers]
    _patch(model)
    n = ane_patch.unpatch_model(model)
    assert n == len(originals)
    for layer, orig in zip(model.transformer.layers, originals, strict=True):
        assert layer.feed_forward is orig


def test_compiled_wrapper_restores_the_ffn_so_the_baseline_stays_clean():
    """bench's gpu_only arm closes over the same blocks, so a leaked swap would give it the ANE."""
    from rowsplit import CompiledSplitBlock

    class Block:
        def __init__(self, ff):
            self.feed_forward = ff

        def __call__(self, x):
            self.seen = self.feed_forward
            return x

    original = object()
    split = object()
    block = Block(original)
    out = CompiledSplitBlock(block, split)("h")

    assert block.seen is split, "the split FFN was not installed for the call"
    assert block.feed_forward is original, "the swap leaked into the baseline"
    assert out == "h"


def test_compiled_wrapper_restores_even_when_the_block_raises():
    from rowsplit import CompiledSplitBlock

    class Boom:
        feed_forward = "original"

        def __call__(self, *a, **k):
            raise RuntimeError("block failed")

    block = Boom()
    with pytest.raises(RuntimeError):
        CompiledSplitBlock(block, "split")(1)
    assert block.feed_forward == "original", "an exception must not leave the split installed"


def test_range_fix_switch_reaches_the_spec():
    assert ane_patch.spec_for(0.5, SEQ, dim=DIM, hidden=HID).range_fix is True
    assert ane_patch.spec_for(0.5, SEQ, range_fix=False, dim=DIM, hidden=HID).range_fix is False
    # name must differ, or two arms would silently share one cached program
    a = ane_patch.spec_for(0.5, SEQ, dim=DIM, hidden=HID).name
    b = ane_patch.spec_for(0.5, SEQ, range_fix=False, dim=DIM, hidden=HID).name
    assert a != b


def test_generate_wrapper_exposes_the_flag_and_parses(monkeypatch):
    import generate_ane

    parser = generate_ane.build_parser()
    generate_ane.add_ane_arguments(parser)
    # mflux's CommandLineParser.parse_args() takes no argv: it reads sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_ane",
            "--prompt",
            "x",
            "--width",
            "1024",
            "--height",
            "1024",
            "--steps",
            "8",
            "--seed",
            "0",
            "--ane-mlp",
        ],
    )
    args = parser.parse_args()
    assert args.ane_mlp is True
    assert args.ane_share == 0.50
    assert args.ane_seq == 4224  # 1024x1024 token count the split is sized for
    assert args.ane_backend == "bridge"
    assert hasattr(args, "output"), "the wrapper saves via the CLI's own --output field"


class _FakeProgram:
    """Stands in for the bridge's wl_fused: consumes the ANE rows, returns the two outputs."""

    def __init__(self, proj, host_read=True):
        self.proj = proj
        self.host_read = host_read
        self.seen_shapes = []
        self.seen_x = None

    def wl_fused(self, x, w1, w3, w2, s1, s3, s2, model):
        self.seen_shapes.append((tuple(x.shape), x.dtype))
        # Reading a traced array from the host is illegal under mx.compile, which is the whole point
        # of the primitive form, so the compile test turns this bookkeeping off.
        if self.host_read:
            self.seen_x = np.array(x)  # float16, exactly what the program received
        # y_rows is [1, mpad, dim] -- the FFN ends at dim -- and h_max_out is [1, mpad, 1].
        y = (x.astype(mx.float32) @ mx.array(self.proj)).astype(mx.float16)
        h_max = mx.array(np.full((1, x.shape[1], 1), 4.0, dtype=np.float16))
        return y, h_max


def _primitive_ffn(monkeypatch, proj, host_read=True):
    import rowsplit
    from mlp_program import MlpSpec
    from rowsplit import BlockWeights, PrimitiveSplitFFN

    calls = []
    real_stream = mx.stream

    def spy(s):
        calls.append(s)
        return real_stream(s)

    # Staging must run on a CPU stream: that forces the Metal commit boundary before the GPU share
    # is encoded, the only form that started the ANE task early (100% overlap vs 56% for a GPU slice).
    monkeypatch.setattr(mx, "stream", spy)
    monkeypatch.setattr(rowsplit, "_cpu_stream", lambda: mx.new_stream(mx.cpu))

    gain = np.arange(1, DIM + 1, dtype=np.float32) / DIM
    w = BlockWeights(
        w1=np.zeros((HID, DIM), dtype=np.int8),
        w3=np.zeros((HID, DIM), dtype=np.int8),
        w2=np.zeros((DIM, HID), dtype=np.int8),
        scales={"s1": np.ones(HID), "s3": np.ones(HID), "s2": np.ones(DIM)},
        gain=gain,
    )
    spec = MlpSpec(mpad=MPAD, dim=DIM, hidden=HID, scale_mode="output", k13_chunks=1, k2_chunks=4, range_fix=True)
    ext = _FakeProgram(proj, host_read)
    ffn = PrimitiveSplitFFN(FeedForward(DIM, HID), ext, object(), spec, w)
    return ffn, ext, gain, calls


def test_primitive_form_stages_the_ane_rows_on_a_cpu_stream(monkeypatch):
    ffn, ext, _gain, calls = _primitive_ffn(monkeypatch, np.zeros((DIM, DIM), np.float32))
    h = _h()
    mx.eval(ffn(h))
    assert len(calls) == 1, f"expected one stream switch for staging, got {len(calls)}"
    assert calls[0].device == mx.cpu, f"staging ran on {calls[0].device}, not the CPU"
    assert ext.seen_shapes == [((1, MPAD, DIM), mx.float16)]


def test_cpu_stream_staging_preserves_the_rows(monkeypatch):
    # Staging on another stream must not change the values the program receives.
    h = _h()
    with mx.stream(mx.new_stream(mx.cpu)):
        staged = h[:, :MPAD].astype(mx.float16)
    mx.eval(staged)
    got = np.array(staged.astype(mx.float32))
    want = np.array(h[:, :MPAD].astype(mx.float16).astype(mx.float32))
    assert np.array_equal(got, want)


def test_primitive_form_gpu_rows_match_the_unpatched_ffn(monkeypatch):
    ffn, _ext, _gain, _calls = _primitive_ffn(monkeypatch, np.zeros((DIM, DIM), np.float32))
    ff = ffn.ff
    h = _h()
    out = ffn(h)
    mx.eval(out)
    ref_gpu = ff(h[:, MPAD:])
    mx.eval(ref_gpu)
    got = np.array(out[:, MPAD:].astype(mx.float32))
    want = np.array(ref_gpu.astype(mx.float32))
    # Not bitwise: the ordering edge can hand the FFN a contiguous input where the slice gave a
    # strided view, and MLX picks a different matmul tiling, so accumulators land 1 bf16 ulp apart.
    assert np.allclose(got, want, rtol=1e-2, atol=1e-3)


def test_primitive_form_restores_h_max_then_gain_in_float32(monkeypatch):
    proj = (np.random.default_rng(7).standard_normal((DIM, DIM)) * DIM**-0.5).astype(np.float32)
    ffn, ext, gain, _calls = _primitive_ffn(monkeypatch, proj)
    h = _h()
    out = ffn(h)
    mx.eval(out)
    expected = (ext.seen_x.astype(np.float64) @ proj.astype(np.float64)) * 4.0 * gain.astype(np.float64)
    got = np.array(out[:, :MPAD].astype(mx.float32)).astype(np.float64)
    # out is cast back to h.dtype (bf16), so the comparison is at bf16 resolution
    assert np.allclose(got, expected, rtol=1e-2, atol=1e-2)
    assert np.isfinite(got).all()


def test_primitive_form_survives_mx_compile(monkeypatch):
    proj = (np.random.default_rng(9).standard_normal((DIM, DIM)) * DIM**-0.5).astype(np.float32)
    ffn, _ext, _gain, _calls = _primitive_ffn(monkeypatch, proj, host_read=False)
    h = _h()
    out = mx.compile(ffn)(h)
    mx.eval(out)
    assert out.shape == h.shape
    assert np.isfinite(np.array(out.astype(mx.float32))).all()
