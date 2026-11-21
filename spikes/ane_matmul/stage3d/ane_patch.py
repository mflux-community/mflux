"""Spike-side wiring: put the row-split fused MLP behind a flag on the real generation path.

Design follows ~/zCode_26_11/drawthings-ane-research/DRAWTHINGS-ANE-FINDINGS.md (liuliu/ccv,
BSD-3-Clause). Written from that description; no source copied.

Kept out of src/mflux on purpose: this is a spike, and the arm must be reachable without touching
the shipped package. generate_ane.py reuses the real Z-Image Turbo CLI parser, model loader and
latent creator, so the only difference from `mflux-generate-z-image-turbo` is this patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

SPIKE = Path(__file__).resolve().parent
if str(SPIKE) not in sys.path:
    sys.path.insert(0, str(SPIKE))

from mlp_program import MlpSpec, plan_split  # noqa: E402
from rowsplit import (  # noqa: E402
    BridgeBackend,
    CallStats,
    patch_transformer as _patch_transformer,
    patch_transformer_compilable,
)

DEFAULT_CACHE = SPIKE / "cache"


def spec_for(share: float, seq_len: int, range_fix: bool = True, **kw) -> MlpSpec:
    """The program spec for a given (share, sequence length). Mpad is always a multiple of 256."""
    m, _ = plan_split(seq_len, share)
    return MlpSpec(mpad=m, scale_mode="output", range_fix=range_fix, **kw)


def make_backend(name: str = "bridge", cache_dir: Path | None = None, shared_surfaces: bool = False):
    if name == "bridge":
        return BridgeBackend(cache_dir or DEFAULT_CACHE, shared_surfaces=shared_surfaces)
    if name == "mirror":
        from bench import MirrorBackend  # numpy emulation: no Core ML, no ANE, for weight-free tests

        return MirrorBackend()
    raise ValueError(f"unknown ane backend {name!r}")


def patch_model(
    model,
    share: float = 0.50,
    seq_len: int = 4224,
    backend_name: str = "bridge",
    range_fix: bool = True,
    cache_dir: Path | None = None,
    shared_surfaces: bool = False,
    stats: CallStats | None = None,
    mode: str = "compiled",
):
    """Replace every main-block FFN with the row-split ANE path. Refiners stay on the GPU.

    seq_len is the token count the split is sized for; the wrapper asserts at call time that the
    real sequence is longer than Mpad, so a wrong guess fails loudly instead of silently putting
    every row on one engine.

    mode="compiled" uses the wl_fused MLX primitive and is the only form that works under mflux's
    mx.compile-wrapped denoise step. mode="bench" uses the threading backend with per-phase timing,
    which is what the standalone benches need and what produced the T2-T4 numbers -- it raises inside
    a compiled graph, so never use it for generation.
    """
    spec = spec_for(share, seq_len, range_fix=range_fix)
    backend = make_backend(backend_name, cache_dir, shared_surfaces)
    # Kept, not discarded: without this handle the real token count and the per-call breakdown are
    # unreachable from the caller, and an image time cannot be attributed to a share. The compute
    # plan lives on the backend and fills in lazily, so it is read after generation, not here.
    stats = stats or CallStats()
    if mode == "compiled":
        n = patch_transformer_compilable(model.transformer, backend, spec)
    elif mode == "bench":
        n = _patch_transformer(model.transformer, backend, spec, stats)
    else:
        raise ValueError(f"unknown patch mode {mode!r}")
    return {
        "spec": spec.name,
        "mpad": spec.mpad,
        "share": share,
        "seq_len_sized_for": seq_len,
        "patch_mode": mode,
        "patched_blocks": n,
        "range_fix": range_fix,
        "backend": backend_name,
        "weight_surfaces": "shared" if shared_surfaces else "per-block",
        "_runtime": {"stats": stats, "backend": backend, "spec_name": spec.name},
    }


def unpatch_model(model):
    """Put the original FeedForward modules back.

    RowSplitFFN holds the module it wraps, so restoring is a reassignment -- NOT building a fresh
    FeedForward, which would replace trained weights with a random initialisation. Each patched
    block also owns a worker thread; restoring drops them so a second arm in the same process
    cannot inherit the first arm's threads or programs.
    """
    import mlx.core as mx

    n = 0
    for layer in model.transformer.layers:
        inner = getattr(layer.feed_forward, "ff", None)
        if inner is not None:
            layer.feed_forward = inner
            n += 1
    mx.eval(model.transformer.parameters())
    return n


__all__ = ["make_backend", "patch_model", "spec_for", "unpatch_model"]
