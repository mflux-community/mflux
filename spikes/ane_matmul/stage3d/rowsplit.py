"""Stage 3d Phase A: row-split MLP. ANE runs the whole MLP for the first `ane_rows` tokens, GPU for the rest.

Everything except the FFN stays as in ZImageTransformerBlock. Flow per block:
  GPU: h = ffn_norm1(x) * scale_mlp                      (full S rows)
  GPU: eval(h[:, :m]) -> numpy fp16 -> ANE thread        (prep: includes the upstream GPU chain it waits on)
  GPU: ff(h[:, m:])  (async_eval, overlaps the ANE call)
  join ANE (wait) -> fp32 epilogue (x32 and optional w2 scale, clip) -> concat on token axis (epilogue)
  GPU: x = x + gate_mlp * ffn_norm2(y)
"""

from __future__ import annotations

import threading
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


_WEIGHT_CACHE: dict[tuple, tuple] = {}


def _fingerprint(src) -> tuple:
    """Cheap content guard for a cache key: shape, dtype and the first element of each array.

    Holding the source arrays to make id() keys safe was the wrong trade: it pinned ~590 MB per
    block, which at 34 real blocks is ~20 GB and drove the machine into 6.3 GB of swap. A hit is
    now verified against the arrays actually passed in, so if an id were ever recycled by a new
    object the entry is simply recomputed. A miss is cheap; a stale hit would be silent.
    """
    out = []
    for a in src:
        shape = tuple(a.shape)
        # .item() rather than np.asarray(...): a 0-d MLX array does not convert through numpy, and
        # these inputs are usually bf16 mx arrays.
        first = float(a[(0,) * len(shape)].item())
        out.append((shape, str(getattr(a, "dtype", "?")), first))
    return tuple(out)


def prepare_block_weights(spec: MlpSpec, w1, w3, w2) -> BlockWeights:
    """w1,w3: [H,D]; w2: [D,H]; any array-like convertible with np.asarray(..., float32).

    Memoised on the source arrays and the weight-bearing spec fields. The quantised codes do not
    depend on mpad, so a share sweep would otherwise re-quantise every block per share: at 34
    blocks and 4 shares that is ~16 GB of duplicate host int8, which is swap territory on 36 GB
    and would invalidate the very runs it is meant to time.
    """
    src = (w1, w3, w2)
    fp = _fingerprint(src)
    key = (id(w1), id(w3), id(w2), spec.weight_dtype, spec.scale_mode, spec.w2_scale, spec.range_fix)
    hit = _WEIGHT_CACHE.get(key)
    if hit is not None and hit[1] == fp:
        return hit[0]
    w1, w3, w2 = (_to_np32(w) for w in src)
    if spec.weight_dtype == "int8":
        (q1, s1), (q3, s3), (q2, s2) = quantize_per_row(w1), quantize_per_row(w3), quantize_per_row(w2)
    else:
        q1, q3, q2 = (w.astype(np.float16) for w in (w1, w3, w2))
        s1, s3, s2 = (np.ones(w.shape[0], dtype=np.float32) for w in (w1, w3, w2))
    bw = BlockWeights(q1, q3, q2, program_scales(spec, s1, s3, s2), epilogue_gain(spec, s2))
    if len(_WEIGHT_CACHE) > 64:  # entries are ~118 MB; bound them across long sweeps
        _WEIGHT_CACHE.clear()
    _WEIGHT_CACHE[key] = (bw, fp)
    return bw


class AneBackend:
    """One ANE call: x [1, Mpad, D] fp16 -> (y/32 [1, Mpad, D] fp16, h_max or None).

    The second value is the per-row max of h that the range fix divides by (spec.range_fix). It
    must be restored in the fp32 GPU epilogue: multiplying it back inside the program overflows
    fp16, which is the failure that made the first real-weight gate fail.
    """

    weights_copied = False

    def run(self, spec: MlpSpec, weights: BlockWeights, x: np.ndarray):
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
        out = self._model(spec).predict(feed)
        return out["y_rows"], out.get("h_max_out")


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

    def __init__(self, cache_dir: str | Path, compiled_dir: str | Path | None = None, shared_surfaces: bool = False):
        import sys

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # False (default) wraps each block's own host weight buffers, which is zero-copy but makes
        # the kernel wire those bytes: probe_wired_surfaces measured wired growth ~1:1 with the int8
        # weight bytes, so 34 blocks cost ~4 GB. True keeps ONE surface set per shape and blits the
        # block's weights in per call: wired stays flat, at ~5.1 ms/call of memcpy.
        self.shared_surfaces = shared_surfaces
        # Durable compile cache, never /tmp: macOS deletes /tmp and the compiled .mlmodelc is
        # the slow part to rebuild (the rule stage 3c learned, cf. aneprobe-tmp-cache memory).
        self.compiled_root = Path(compiled_dir or (Path.home() / "Library/Caches/mflux/ane_wl3d"))
        self.compiled_root.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ane_linear"))
        import _ane_linear as ext  # lazy: the numpy-only tests must not need the bridge

        self.ext = ext
        self._models: dict[str, object] = {}
        self._outs: dict[str, np.ndarray] = {}
        self._surfaces: list[np.ndarray] | None = None
        self._surface_lock = threading.Lock()
        self.plan_reports: dict[str, str] = {}

    def _prepare(self, spec: MlpSpec):
        if spec.name in self._models:
            return self._models[spec.name], self._outs[spec.name]
        M, D, H = spec.mpad, spec.dim, spec.hidden
        pkg = self.cache_dir / f"{spec.name}.mlpackage"
        if not pkg.exists():
            convert_to_mlpackage(spec, str(self.cache_dir))
        compiled = self.compiled_root / spec.name
        # The weight dtype must be declared, not assumed: build_mil_program emits fp16 weight
        # inputs under --weight-dtype fp16, and wrapping those as int8 mislabels the buffer.
        w_i8 = 1 if spec.weight_dtype == "int8" else 0
        out_names = ["y_rows"] + (["h_max_out"] if spec.range_fix else [])
        out_shapes = [[1, M, D]] + ([[1, M, 1]] if spec.range_fix else [])
        model = self.ext.wl_fused_load(
            str(pkg),
            str(compiled),
            ["x_rows", "w1", "w3", "w2", "s1", "s3", "s2"],
            [[1, M, D], [H, D], [H, D], [D, H], [H], [H], [D]],
            [0, w_i8, w_i8, w_i8, 0, 0, 0],
            out_names,
            out_shapes,
            [0] * len(out_names),
        )
        # After the load, because wl_fused_load is what produces the compiled .mlmodelc.
        self.plan_reports.setdefault(spec.name, compute_plan_report(compiled))
        # One ANE worker thread serialises calls, so reusable outputs per spec are safe.
        outs = [np.zeros(s, dtype=np.float16) for s in out_shapes]
        self._models[spec.name] = model
        self._outs[spec.name] = outs
        return model, outs

    def _ensure_surfaces(self, weights):
        """One weight surface set for every block, so the kernel wires ~118 MB once, not 34 times."""
        if self._surfaces is None:
            self._surfaces = [
                np.empty_like(weights.w1),
                np.empty_like(weights.w3),
                np.empty_like(weights.w2),
                np.empty_like(weights.scales["s1"]),
                np.empty_like(weights.scales["s3"]),
                np.empty_like(weights.scales["s2"]),
            ]
        for dst, src in zip(
            self._surfaces,
            (weights.w1, weights.w3, weights.w2, weights.scales["s1"], weights.scales["s3"], weights.scales["s2"]),
            strict=True,
        ):
            np.copyto(dst, src)
        return self._surfaces

    def run(self, spec, weights, x):
        model, outs = self._prepare(spec)
        x_in = np.ascontiguousarray(x, dtype=np.float16)
        # No dtype is forced on the weights: prepare_block_weights already produced int8 codes or
        # fp16 according to spec.weight_dtype, and the loader declares the matching type.
        if self.shared_surfaces:
            # One surface set is shared by every block, and each RowSplitFFN owns its own worker
            # thread, so a second block could blit over the first one's weights mid-call. Inside a
            # chain that cannot happen, but the cost of not relying on it is an uncontended lock.
            with self._surface_lock:
                ins = [x_in, *self._ensure_surfaces(weights)]
                self.ext.wl_fused_predict(model, [a.ctypes.data for a in ins], [a.ctypes.data for a in outs])
        else:
            ins = [
                x_in,
                np.ascontiguousarray(weights.w1),
                np.ascontiguousarray(weights.w3),
                np.ascontiguousarray(weights.w2),
                np.ascontiguousarray(weights.scales["s1"], dtype=np.float16),
                np.ascontiguousarray(weights.scales["s3"], dtype=np.float16),
                np.ascontiguousarray(weights.scales["s2"], dtype=np.float16),
            ]
            self.ext.wl_fused_predict(model, [a.ctypes.data for a in ins], [a.ctypes.data for a in outs])
        err = model.last_error()
        if err:
            raise RuntimeError(f"bridge predict failed: {err}")
        # outs[1] is h_max under the range fix; the caller restores it in fp32, outside the program.
        return outs[0], (outs[1] if len(outs) > 1 else None)


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
    # Z-Image's main layers run on the unified sequence, so S moves with prompt length, and an image
    # timing means nothing unless the real S -- hence the real share -- is recorded.
    seen_seq: set[int] = field(default_factory=set)

    def summary_ms(self) -> dict[str, float]:
        # medians over the timing lists only; seen_seq is a set of token counts, not milliseconds
        return {k: 1e3 * float(np.median(v)) if v else 0.0 for k, v in vars(self).items() if isinstance(v, list)}

    def seq_seen(self) -> list[int]:
        return sorted(self.seen_seq)


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
        self.stats.seen_seq.add(int(h.shape[1]))
        h_ane = h[:, :m].astype(mx.float16)

        t0 = time.perf_counter()
        x_np = np.array(h_ane)  # forces eval of everything upstream, then copies to host
        t1 = time.perf_counter()
        fut = self._pool.submit(self.backend.run, self.spec, self.weights, x_np)

        y_gpu = self.ff(h[:, m:])
        mx.async_eval(y_gpu)

        t2 = time.perf_counter()
        y_np, h_max = fut.result()
        t3 = time.perf_counter()

        y_ane = mx.array(np.asarray(y_np, dtype=np.float16)).astype(mx.float32)
        if h_max is not None:
            # The range fix divides h by this per-row max inside the program. Restoring it there is
            # what overflows fp16 (measured 5.06e5 against a 65504 ceiling), so it comes back here,
            # in fp32, on the ANE's own rows only -- the GPU rows were never normalised.
            y_ane = y_ane * mx.array(np.asarray(h_max, dtype=np.float32))
        y_ane = y_ane * self._gain
        limit = float(mx.finfo(mx.float16).max)
        y_ane = mx.clip(y_ane, -limit, limit).astype(h.dtype)  # same clip as ComputePrecision.to_stream
        out = mx.concatenate([y_ane, y_gpu.astype(h.dtype)], axis=1)
        mx.eval(out)
        t4 = time.perf_counter()

        self.stats.prep.append(t1 - t0)
        self.stats.wait.append(t3 - t2)
        self.stats.epilogue.append(t4 - t3)
        return out


class CompiledSplitBlock:
    """Runs the real block with its feed_forward swapped for the primitive split, for one call only.

    bench's gpu_only arm closes over the same block objects, so a permanent in-place patch would give
    the baseline the ANE too and quietly destroy the comparison. Swapping per call keeps the arms
    independent, and the block's own forward still comes from transformer_block.py, so nothing here
    re-implements it the way RowSplitBlock does.
    """

    def __init__(self, block, split_ffn):
        self.block, self.split = block, split_ffn

    def __call__(self, *args, **kwargs):
        orig = self.block.feed_forward
        self.block.feed_forward = self.split
        try:
            return self.block(*args, **kwargs)
        finally:
            self.block.feed_forward = orig


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


class PrimitiveSplitFFN:
    """Row split that is legal inside mx.compile: calls the wl_fused primitive, never the host.

    RowSplitFFN exists for the standalone benches, which time the phases by synchronising. That is
    fatal in mflux's denoise step, which runs under mx.compile, so generation uses this form: pure
    MLX ops, so MLX schedules the ANE call itself.

    The rowmax restore is done in float32 on purpose. In fp16 it overflows -- real h_max reaches 5064
    and the restored product hit 5.06e5 against a 65504 ceiling, which is the original bug.
    """

    def __init__(self, ff, ext, model, spec: MlpSpec, weights: BlockWeights):
        import mlx.core as mx

        self.ff, self.ext, self.model, self.spec = ff, ext, model, spec
        self._mxw = [
            mx.array(weights.w1),
            mx.array(weights.w3),
            mx.array(weights.w2),
            mx.array(weights.scales["s1"]),
            mx.array(weights.scales["s3"]),
            mx.array(weights.scales["s2"]),
        ]
        self._gain = mx.array(weights.gain)  # fp32 per-channel

    def __call__(self, h):
        import mlx.core as mx

        m = self.spec.mpad
        assert h.shape[0] == 1 and h.shape[1] > m, "row split needs batch 1 and S > Mpad"
        # The ANE branch's staging kernel must be encoded before the GPU share, or MLX puts the
        # cross-stream fence after it: 56-58% overlap against 100% for an ordered fork. depends()
        # orders them while keeping the cast on the GPU. Casting on a CPU stream instead hands Core
        # ML a host buffer, which wires ~31 MB per block and paged 2.1 GB at rung 34.
        h_ane = h[:, :m].astype(mx.float16)
        h_gpu = mx.depends([h[:, m:]], [h_ane])[0]
        y, h_max = self.ext.wl_fused(h_ane, *self._mxw, self.model)
        y_gpu = self.ff(h_gpu)  # enqueue the GPU share before the join, or the join stalls it
        y_ane = y.astype(mx.float32) * h_max.astype(mx.float32) * self._gain
        return mx.concatenate([y_ane.astype(h.dtype), y_gpu.astype(h.dtype)], axis=1)


def patch_transformer_compilable(transformer, backend: BridgeBackend, spec: MlpSpec):
    """Swap each main block's FFN for the primitive form. Returns the number patched.

    Attribute swap, same reason as patch_transformer: the block's own forward stays in play, so this
    cannot drift from transformer_block.py. No worker threads are created, because nothing waits on
    a host result.
    """
    ext = backend.ext
    for layer in transformer.layers:
        ff = layer.feed_forward
        w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
        model, _outs = backend._prepare(spec)
        layer.feed_forward = PrimitiveSplitFFN(ff, ext, model, spec, w)
    return len(transformer.layers)


def patch_transformer(transformer, backend: AneBackend, spec: MlpSpec, stats: CallStats):
    """Swap every main layer's FFN for a RowSplitFFN. Real weights: ask before running.

    Use this for the standalone benches, which time the phases and therefore must synchronise. Use
    patch_transformer_compilable for generation, where mflux wraps the step in mx.compile.

    Replaces the attribute rather than wrapping the block, because ZImageTransformerBlock calls
    `self.feed_forward(self.ffn_norm1(x) * scale_mlp)` with one positional argument: the attribute
    swap inherits the block's real forward, so the generation path cannot drift from
    transformer_block.py the way a re-implementation would. Refiners are left untouched -- the
    split is sized for the main sequence.

    Returns the number of patched layers; one ANE worker thread is created per patched block.
    """
    for layer in transformer.layers:
        ff = layer.feed_forward
        w = prepare_block_weights(spec, ff.w1.weight, ff.w3.weight, ff.w2.weight)
        layer.feed_forward = RowSplitFFN(ff, backend, spec, w, stats)
    return len(transformer.layers)
