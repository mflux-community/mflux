from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config


class StepCache:
    """Model-agnostic TeaCache-style step reuse for a denoise loop.

    On a skipped step the loop reuses the previous step's model output (the noise
    or velocity prediction, after any guidance) instead of running the model, and
    the scheduler still takes its normal step with it. Which steps are skipped is
    decided once, before the loop:

    - Each step gets a scalar-or-vector *signal*. By default that is the step's
      sigma, which every flow-match scheduler exposes, so any model can opt in
      without model-specific code. A model can pass ``signal_fn`` to score steps
      with a richer signal (for example its timestep-embedding MLP, as in
      Liu et al., "Timestep Embedding Tells: It's All You Need for Accelerating
      DiT-based Diffusion Models", ICLR 2025).
    - The ``ratio``-fraction of steps with the smallest step-to-step signal change
      are skipped, inside a protected window: the first and last 10% of the run
      are always computed, and runs shorter than ``MIN_STEPS`` skip nothing.

    The signal depends only on the schedule, so selection is deterministic for a
    given (steps, resolution, ratio) and costs one batched evaluation per run.

    The skip set is derived from ``config.init_time_step`` and
    ``config.num_inference_steps`` rather than by iterating ``config.time_steps``:
    that property is the loop's shared tqdm progress bar, and consuming it early
    would disable the bar for the real loop.

    Usage in a model's denoise loop::

        step_cache = StepCache.for_run(config, ratio=step_cache_ratio)
        for t in config.time_steps:
            noise = step_cache.reuse(t)
            if noise is None:
                noise = ...  # run the model (and any guidance pass)
                step_cache.store(noise)
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
    """

    MIN_STEPS = 10
    PROTECTED_FRACTION = 0.1

    def __init__(self, skip_steps: frozenset[int]):
        self.skip_steps = skip_steps
        self._previous: mx.array | None = None

    @staticmethod
    def for_run(
        config: Config,
        ratio: float | None,
        signal_fn: Callable[[mx.array], mx.array] | None = None,
    ) -> StepCache:
        """Build the per-run cache; ``ratio=None`` returns an inactive cache."""
        if ratio is None:
            return StepCache(frozenset())
        steps = list(range(config.init_time_step, config.num_inference_steps))
        sigmas = config.scheduler.sigmas
        return StepCache(StepCache.select_skip_steps(steps, sigmas, ratio, signal_fn))

    @staticmethod
    def select_skip_steps(
        steps: list[int],
        sigmas: mx.array,
        ratio: float,
        signal_fn: Callable[[mx.array], mx.array] | None = None,
    ) -> frozenset[int]:
        if not 0 < ratio < 1:
            raise ValueError(f"step cache ratio must be within (0, 1), got {ratio}")
        total = len(steps)
        if total < StepCache.MIN_STEPS:
            return frozenset()
        step_sigmas = sigmas[mx.array(steps)].astype(mx.float32)
        signals = signal_fn(step_sigmas) if signal_fn is not None else step_sigmas
        signals = signals.astype(mx.float32).reshape(total, -1)
        # distance between consecutive steps; index i compares step i with step i-1
        diffs = mx.sqrt(mx.sum(mx.square(signals[1:] - signals[:-1]), axis=-1))
        diffs = mx.concatenate([mx.array([float("inf")]), diffs])
        diff_values = diffs.tolist()
        protected = int(total * StepCache.PROTECTED_FRACTION)
        eligible = list(range(max(protected, 1), total - protected))
        if not eligible:
            return frozenset()
        count = min(round(ratio * total), len(eligible))
        chosen = sorted(eligible, key=lambda i: diff_values[i])[:count]
        return frozenset(steps[i] for i in chosen)

    @staticmethod
    def generation_parameters(ratio: float | None) -> dict:
        """Metadata for the image: a reuse run is a different image, so record the ratio.

        Only runs that asked for reuse gain the key, so every other image's metadata is
        unchanged; --config-from-metadata reads it back.
        """
        return {"step_cache_ratio": ratio} if ratio is not None else {}

    @property
    def active(self) -> bool:
        return bool(self.skip_steps)

    def reuse(self, t: int) -> mx.array | None:
        """The stored model output when step ``t`` is skipped, else ``None``."""
        if t in self.skip_steps and self._previous is not None:
            return self._previous
        return None

    def store(self, output: mx.array) -> None:
        """Remember the model output of a computed step for later reuse."""
        self._previous = output
