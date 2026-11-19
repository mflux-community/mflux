"""Stage 3d Phase A: row-split MLP. ANE runs the whole MLP for the first `ane_rows` tokens, GPU for the rest.

Everything except the FFN stays as in ZImageTransformerBlock. Flow per block:
  GPU: h = ffn_norm1(x) * scale_mlp                      (full S rows)
  GPU: eval(h[:, :m]) -> numpy fp16 -> ANE thread        (prep: includes the upstream GPU chain it waits on)
  GPU: ff(h[:, m:])  (async_eval, overlaps the ANE call)
  join ANE (wait) -> fp32 epilogue (x32 and optional w2 scale, clip) -> concat on token axis (epilogue)
  GPU: x = x + gate_mlp * ffn_norm2(y)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mlp_program import (
    MlpSpec,
    convert_to_mlpackage,
    epilogue_gain,
    program_scales,
    quantize_per_row,
)


@dataclass
class BlockWeights:
    w1: np.ndarray
    w3: np.ndarray
    w2: np.ndarray
    scales: dict[str, np.ndarray]
    gain: np.ndarray  # fp32 [D] epilogue gain


def _to_np32(w) -> np.ndarray:
    if isinstance(w, np.ndarray):
        return w.astype(np.float32)
    import mlx.core as mx  # bf16 mx arrays have no numpy buffer; widen first

    return np.array(w.astype(mx.float32))


def prepare_block_weights(spec: MlpSpec, w1, w3, w2) -> BlockWeights:
    """w1,w3: [H,D]; w2: [D,H]; any array-like convertible with np.asarray(..., float32)."""
    w1, w3, w2 = (_to_np32(w) for w in (w1, w3, w2))
    if spec.weight_dtype == "int8":
        (q1, s1), (q3, s3), (q2, s2) = quantize_per_row(w1), quantize_per_row(w3), quantize_per_row(w2)
    else:
        q1, q3, q2 = (w.astype(np.float16) for w in (w1, w3, w2))
        s1, s3, s2 = (np.ones(w.shape[0], dtype=np.float32) for w in (w1, w3, w2))
    return BlockWeights(q1, q3, q2, program_scales(spec, s1, s3, s2), epilogue_gain(spec, s2))


class AneBackend:
    """One ANE call: x [1, Mpad, D] fp16 -> y/32 [1, Mpad, D] fp16."""

    weights_copied = False

    def run(self, spec: MlpSpec, weights: BlockWeights, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CoreMLToolsBackend(AneBackend):
    """Reference backend via coremltools predict. Correct, but copies ~118 MB of weights per call,
    so its timings are NOT gate-valid. Use for T1 (correctness) and smoke tests only."""

    weights_copied = True

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, object] = {}
        self.plan_reports: dict[str, str] = {}

    def _model(self, spec: MlpSpec):
        import coremltools as ct

        if spec.name not in self._models:
            path = self.cache_dir / f"{spec.name}.mlpackage"
            if not path.exists():
                convert_to_mlpackage(spec, str(self.cache_dir))
            self._models[spec.name] = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.plan_reports[spec.name] = compute_plan_report(path)
        return self._models[spec.name]

    def run(self, spec, weights, x):
        feed = {
            "x_rows": x,
            "w1": weights.w1,
            "w3": weights.w3,
            "w2": weights.w2,
            **weights.scales,
        }
        return self._model(spec).predict(feed)["y_rows"]


class BridgeBackend(AneBackend):
    """The 3c weightless bridge (f16ba7bf), extended in 3d with a fused multi-input call.

    The program holds no baked weights: w1/w3/w2 arrive as int8 graph inputs and the scales as
    fp16, all wrapped around the caller's numpy buffers with initWithDataPointer, and the output
    wraps a reusable buffer. No per-call weight copy and no per-call Core ML setup, which is the
    only way these timings can be gate-valid (weights_copied stays False from the base class).

    The predict releases the GIL, so the main thread can enqueue the GPU's rows while the ANE
    runs its own -- without that, the row split's whole premise (both engines working at once)
    would not even be measured.
    """

    def __init__(self, cache_dir: str | Path, compiled_dir: str | Path | None = None):
        import sys

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Durable compile cache, never /tmp: macOS deletes /tmp and the compiled .mlmodelc is
        # the slow part to rebuild (the rule stage 3c learned, cf. aneprobe-tmp-cache memory).
        self.compiled_root = Path(compiled_dir or (Path.home() / "Library/Caches/mflux/ane_wl3d"))
        self.compiled_root.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ane_linear"))
        import _ane_linear as ext  # lazy: the numpy-only tests must not need the bridge

        self.ext = ext
        self._models: dict[str, object] = {}
        self._outs: dict[str, np.ndarray] = {}
        self.plan_reports: dict[str, str] = {}

    def _prepare(self, spec: MlpSpec):
        if spec.name in self._models:
            return self._models[spec.name], self._outs[spec.name]
        M, D, H = spec.mpad, spec.dim, spec.hidden
        pkg = self.cache_dir / f"{spec.name}.mlpackage"
        if not pkg.exists():
            convert_to_mlpackage(spec, str(self.cache_dir))
        compiled = self.compiled_root / spec.name
        model = self.ext.wl_fused_load(
            str(pkg),
            str(compiled),
            ["x_rows", "w1", "w3", "w2", "s1", "s3", "s2"],
            [[1, M, D], [H, D], [H, D], [D, H], [H], [H], [D]],
            [0, 1, 1, 1, 0, 0, 0],
            "y_rows",
            [1, M, D],
        )
        # After the load, because wl_fused_load is what produces the compiled .mlmodelc.
        self.plan_reports.setdefault(spec.name, compute_plan_report(compiled))
        # One ANE worker thread serialises calls, so one reusable output per spec is safe.
        out = np.zeros((1, M, D), dtype=np.float16)
        self._models[spec.name] = model
        self._outs[spec.name] = out
        return model, out

    def run(self, spec, weights, x):
        model, out = self._prepare(spec)
        ins = [
            np.ascontiguousarray(x, dtype=np.float16),
            np.ascontiguousarray(weights.w1, dtype=np.int8),
            np.ascontiguousarray(weights.w3, dtype=np.int8),
            np.ascontiguousarray(weights.w2, dtype=np.int8),
            np.ascontiguousarray(weights.scales["s1"], dtype=np.float16),
            np.ascontiguousarray(weights.scales["s3"], dtype=np.float16),
            np.ascontiguousarray(weights.scales["s2"], dtype=np.float16),
        ]
        self.ext.wl_fused_predict(model, [a.ctypes.data for a in ins], out.ctypes.data)
        err = model.last_error()
        if err:
            raise RuntimeError(f"bridge predict failed: {err}")
        return out


def compute_plan_report(model_dir: Path) -> str:
    """Open question 1: which compute device does each op get? (macOS 14.4+ MLComputePlan).

    model_dir must be a COMPILED .mlmodelc. Handing this a .mlpackage aborts the whole process:
    MLComputePlan tries to read <pkg>/coremldata.bin, fails, and throws a std::ios_base::failure
    that no Python try/except can catch (measured 2026-11-20 -- it killed the first 3d probe).
    The suffix check below is the only safe guard, so callers pass the bridge's compiled cache.
    """
    model_dir = Path(model_dir)
    if model_dir.suffix == ".mlpackage":
        return "compute plan skipped: needs a compiled .mlmodelc; a .mlpackage aborts the process"
    try:
        import coremltools as ct

        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            str(model_dir), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        main = plan.model_structure.program.functions["main"]
        counts: dict[str, int] = {}
        fallbacks = []
        for op in main.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = type(usage.preferred_compute_device).__name__ if usage else "const/none"
            counts[dev] = counts.get(dev, 0) + 1
            if usage and "NeuralEngine" not in dev and op.operator_name not in ("const", "constexpr_cast"):
                fallbacks.append(f"{op.operator_name}->{dev}")
        return f"preferred devices: {counts}; non-ANE ops: {fallbacks or 'none'}"
    except Exception as exc:  # noqa: BLE001 -- report-only, never fail the run
        return f"compute plan unavailable: {exc}"


@dataclass
class CallStats:
    prep: list[float] = field(default_factory=list)
    wait: list[float] = field(default_factory=list)
    epilogue: list[float] = field(default_factory=list)

    def summary_ms(self) -> dict[str, float]:
        return {k: 1e3 * float(np.median(v)) if v else 0.0 for k, v in vars(self).items()}


class RowSplitFFN:
    """FFN for a [1, S, D] activation: first Mpad rows on the ANE, the rest on the GPU, concat on tokens."""

    def __init__(self, ff, backend: AneBackend, spec: MlpSpec, weights: BlockWeights, stats: CallStats):
        import mlx.core as mx

        self.ff, self.backend, self.spec, self.weights, self.stats = ff, backend, spec, weights, stats
        self._gain = mx.array(weights.gain)
        self._pool = ThreadPoolExecutor(max_workers=1)

    def __call__(self, h):
        import mlx.core as mx

        m = self.spec.mpad
        assert h.shape[0] == 1 and h.shape[1] > m, "row split needs batch 1 and S > Mpad"
        h_ane = h[:, :m].astype(mx.float16)

        t0 = time.perf_counter()
        x_np = np.array(h_ane)  # forces eval of everything upstream, then copies to host
        t1 = time.perf_counter()
        fut = self._pool.submit(self.backend.run, self.spec, self.weights, x_np)

        y_gpu = self.ff(h[:, m:])
        mx.async_eval(y_gpu)

        t2 = time.perf_counter()
        y_np = fut.result()
        t3 = time.perf_counter()

        y_ane = mx.array(np.asarray(y_np, dtype=np.float16)).astype(mx.float32) * self._gain
        limit = float(mx.finfo(mx.float16).max)
        y_ane = mx.clip(y_ane, -limit, limit).astype(h.dtype)  # same clip as ComputePrecision.to_stream
        out = mx.concatenate([y_ane, y_gpu.astype(h.dtype)], axis=1)
        mx.eval(out)
        t4 = time.perf_counter()

        self.stats.prep.append(t1 - t0)
        self.stats.wait.append(t3 - t2)
        self.stats.epilogue.append(t4 - t3)
        return out


class RowSplitBlock:
    """Wraps a ZImageTransformerBlock; the FFN is a RowSplitFFN, everything else is unchanged."""

    def __init__(self, block, split_ffn: RowSplitFFN):
        self.block, self.split_ffn = block, split_ffn

    def __call__(self, x, attn_mask, freqs_cis, t_emb):
        import mlx.core as mx

        b = self.block
        modulation = mx.expand_dims(b.adaLN_modulation[0](t_emb), axis=1)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation, 4, axis=2)
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        gate_msa, gate_mlp = mx.tanh(gate_msa), mx.tanh(gate_mlp)

        attn_out = b.attention(b.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + gate_msa * b.attention_norm2(attn_out)
        ffn_out = self.split_ffn(b.ffn_norm1(x) * scale_mlp)
        return x + gate_mlp * b.ffn_norm2(ffn_out)


def patch_transformer(transformer, backend: AneBackend, spec: MlpSpec, stats: CallStats):
    """Swap every main layer for a RowSplitBlock (real-weights path; ask before running with real weights)."""
    for i, layer in enumerate(transformer.layers):
        ff = layer.feed_forward
        w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
        transformer.layers[i] = RowSplitBlock(layer, RowSplitFFN(ff, backend, spec, w, stats))
