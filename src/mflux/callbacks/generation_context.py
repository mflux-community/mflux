from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import mlx.core as mx
import PIL.Image
import tqdm

if TYPE_CHECKING:
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.common.config.config import Config


class GenerationContext:
    def __init__(
        self,
        registry: CallbackRegistry,
        seed: int,
        prompt: str,
        config: Config,
    ):
        self._registry = registry
        self._seed = seed
        self._prompt = prompt
        self._config = config

    def before_loop(
        self,
        latents: mx.array,
        *,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
        control_images: list[PIL.Image.Image] | None = None,
    ) -> None:
        for subscriber in self._registry.before_loop_callbacks():
            subscriber.call_before_loop(
                seed=self._seed,
                prompt=self._prompt,
                latents=latents,
                config=self._config,
                canny_image=canny_image,
                depth_image=depth_image,
                control_images=control_images,
            )

    def in_loop(
        self,
        t: int,
        latents: mx.array,
        time_steps: tqdm = None,
        denoised: mx.array | None = None,
    ) -> None:
        time_steps = time_steps or self._config.time_steps
        for subscriber in self._registry.in_loop_callbacks():
            # Opt-in by signature: third-party callbacks with the fixed, pre-existing
            # `call_in_loop` signature (no `denoised`, no **kwargs) must keep working untouched.
            extra = {"denoised": denoised} if GenerationContext._accepts_denoised(subscriber) else {}
            subscriber.call_in_loop(
                t=t,
                seed=self._seed,
                prompt=self._prompt,
                latents=latents,
                config=self._config,
                time_steps=time_steps,
                **extra,
            )

    def after_loop(self, latents: mx.array) -> None:
        for subscriber in self._registry.after_loop_callbacks():
            subscriber.call_after_loop(
                seed=self._seed,
                prompt=self._prompt,
                latents=latents,
                config=self._config,
            )

    def interruption(self, t: int, latents: mx.array, time_steps: tqdm = None) -> None:
        time_steps = time_steps or self._config.time_steps
        for subscriber in self._registry.interrupt_callbacks():
            subscriber.call_interrupt(
                t=t,
                seed=self._seed,
                prompt=self._prompt,
                latents=latents,
                config=self._config,
                time_steps=time_steps,
            )

    @staticmethod
    def _accepts_denoised(subscriber) -> bool:
        parameters = inspect.signature(subscriber.call_in_loop).parameters.values()
        return any(p.name == "denoised" or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
