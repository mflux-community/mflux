import mlx.core as mx
import pytest

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig


class _Fixtures:
    @staticmethod
    def config() -> Config:
        return Config(
            width=256,
            height=256,
            guidance=1.0,
            scheduler="linear",
            model_config=ModelConfig.flux2_klein_4b(),
            num_inference_steps=1,
        )


class _Decorators:
    @staticmethod
    def passthrough_without_wraps(func):
        # Deliberately no functools.wraps: the wrapper's own signature is what
        # inspect.signature sees, not the wrapped function's.
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


class _DeclaringCallback:
    def __init__(self) -> None:
        self.received_denoised = "not_called"

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, denoised):
        self.received_denoised = denoised


class _KeywordOnlyDeclaringCallback:
    def __init__(self) -> None:
        self.received_denoised = "not_called"

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, *, denoised=None):
        self.received_denoised = denoised


class _PositionalOnlyDeclaringCallback:
    def __init__(self) -> None:
        self.call_count = 0
        self.received_denoised = "not_opted_in"
        self.received_kwargs = None

    def call_in_loop(self, denoised="not_opted_in", /, *, t, seed, prompt, latents, config, time_steps):
        self.call_count += 1
        self.received_denoised = denoised
        self.received_kwargs = {
            "t": t,
            "seed": seed,
            "prompt": prompt,
            "latents": latents,
            "config": config,
            "time_steps": time_steps,
        }


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


class _DecoratedFixedSignatureCallback:
    def __init__(self) -> None:
        self.received_latents = None

    @_Decorators.passthrough_without_wraps
    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        self.received_latents = latents


class _UninspectableCallInLoop:
    # A small callable whose __signature__ raises, so inspect.signature(...)
    # cannot be used to decide whether it opts in to `denoised`.
    def __init__(self, owner: "_UninspectableSignatureCallback") -> None:
        self._owner = owner

    def __call__(self, **kwargs) -> None:
        self._owner.received_kwargs = kwargs

    @property
    def __signature__(self):
        raise ValueError("signature not available")


class _UninspectableSignatureCallback:
    def __init__(self) -> None:
        self.received_kwargs: dict | None = None
        self.call_in_loop = _UninspectableCallInLoop(self)


@pytest.mark.fast
def test_callback_declaring_denoised_receives_the_prediction():
    registry = CallbackRegistry()
    callback = _DeclaringCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_Fixtures.config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.received_denoised is denoised


@pytest.mark.fast
def test_callback_declaring_denoised_keyword_only_receives_the_prediction():
    registry = CallbackRegistry()
    callback = _KeywordOnlyDeclaringCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_Fixtures.config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.received_denoised is denoised


@pytest.mark.fast
def test_positional_only_denoised_does_not_opt_in():
    registry = CallbackRegistry()
    callback = _PositionalOnlyDeclaringCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_Fixtures.config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.call_count == 1
    assert callback.received_denoised == "not_opted_in"
    assert callback.received_kwargs["latents"] is latents


@pytest.mark.fast
def test_callback_with_the_fixed_signature_still_runs_and_gets_no_extra_argument():
    registry = CallbackRegistry()
    callback = _FixedSignatureCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_Fixtures.config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.received_latents is latents


@pytest.mark.fast
def test_bare_var_keywords_callback_does_not_receive_the_prediction():
    registry = CallbackRegistry()
    callback = _VarKeywordsCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_Fixtures.config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert "denoised" not in callback.received_kwargs


@pytest.mark.fast
def test_fixed_signature_callback_behind_a_decorator_without_wraps_still_runs():
    registry = CallbackRegistry()
    callback = _DecoratedFixedSignatureCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_Fixtures.config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.received_latents is latents


@pytest.mark.fast
def test_callback_whose_signature_cannot_be_inspected_still_runs_without_the_prediction():
    registry = CallbackRegistry()
    callback = _UninspectableSignatureCallback()
    registry.register(callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_Fixtures.config())

    latents = mx.zeros((1, 4))
    denoised = mx.ones((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker", denoised=denoised)

    assert callback.received_kwargs is not None
    assert callback.received_kwargs["latents"] is latents
    assert "denoised" not in callback.received_kwargs


@pytest.mark.fast
def test_in_loop_without_a_prediction_passes_none_to_callbacks_that_ask():
    registry = CallbackRegistry()
    declaring_callback = _DeclaringCallback()
    fixed_callback = _FixedSignatureCallback()
    registry.register(declaring_callback)
    registry.register(fixed_callback)
    ctx = registry.start(seed=1, prompt="a cat", config=_Fixtures.config())

    latents = mx.zeros((1, 4))
    ctx.in_loop(0, latents, time_steps="step-tracker")

    assert declaring_callback.received_denoised is None
    assert fixed_callback.received_latents is latents
