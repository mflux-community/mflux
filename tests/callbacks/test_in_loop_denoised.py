import mlx.core as mx
import pytest

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig


def _config() -> Config:
    return Config(
        width=256,
        height=256,
        guidance=1.0,
        scheduler="linear",
        model_config=ModelConfig.flux2_klein_4b(),
        num_inference_steps=1,
    )


class _DeclaringCallback:
    def __init__(self) -> None:
        self.received_denoised = "not_called"

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, denoised=None):
        self.received_denoised = denoised


class _FixedSignatureCallback:
    def __init__(self) -> None:
        self.received_latents = None

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        self.received_latents = latents


class _VarKeywordsCallback:
    def __init__(self) -> None:
        self.received_kwargs = None

    def call_in_loop(self, **kwargs):
        self.received_kwargs = kwargs


@pytest.mark.fast
def test_callback_declaring_denoised_receives_the_prediction():
    registry = CallbackRegistry()
    callback = _DeclaringCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.received_denoised is denoised


@pytest.mark.fast
def test_callback_with_the_fixed_signature_still_runs_and_gets_no_extra_argument():
    registry = CallbackRegistry()
    callback = _FixedSignatureCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.received_latents is latents


@pytest.mark.fast
def test_callback_accepting_var_keywords_receives_the_prediction():
    registry = CallbackRegistry()
    callback = _VarKeywordsCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.received_kwargs["denoised"] is denoised


@pytest.mark.fast
def test_in_loop_without_a_prediction_passes_none_to_callbacks_that_ask():
    registry = CallbackRegistry()
    declaring_callback = _DeclaringCallback()
    fixed_callback = _FixedSignatureCallback()
    registry.register(declaring_callback)
    registry.register(fixed_callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_config())

    latents = mx.zeros((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker")

    assert declaring_callback.received_denoised is None
    assert fixed_callback.received_latents is latents
