# mflux-upscale-controlnet registered --guidance and --scheduler through
# add_image_generator_arguments but never passed either to generate_image, so whatever the
# user typed was replaced by Flux1Controlnet's own defaults (4.0 / "linear") without a word.

import sys

import pytest

from mflux.models.flux.cli import flux_upscale


@pytest.fixture
def run(monkeypatch, tmp_path):
    captured = {}

    class StubModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate_image(self, **kwargs):
            captured.update(kwargs)
            raise SystemExit(0)

    def go(*argv: str):
        monkeypatch.setattr(flux_upscale, "Flux1Controlnet", StubModel)
        monkeypatch.setattr(flux_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
        monkeypatch.setattr(flux_upscale.DimensionResolver, "resolve", staticmethod(lambda **kwargs: (1024, 1024)))
        image = tmp_path / "source.png"
        image.touch()
        monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "sharpen", "--controlnet-image-path", str(image), *argv])  # fmt: off
        with pytest.raises(SystemExit):
            flux_upscale.main()
        return captured

    return go


@pytest.mark.fast
def test_guidance_reaches_the_model(run):
    assert run("--guidance", "6.5")["guidance"] == pytest.approx(6.5)


@pytest.mark.fast
def test_scheduler_reaches_the_model(run):
    assert run("--scheduler", "some_package.some_module.FooScheduler")["scheduler"] == "some_package.some_module.FooScheduler"  # fmt: off


@pytest.mark.fast
def test_omitting_guidance_keeps_the_value_the_upscaler_always_used(run):
    # Not ui_defaults.GUIDANCE_SCALE (3.5): wiring the flag up must not quietly restyle
    # every existing invocation that never passed it.
    assert run()["guidance"] == pytest.approx(4.0)


@pytest.mark.fast
def test_negative_prompt_warns_instead_of_passing_silently(run, recwarn):
    run("--negative-prompt", "blurry")
    assert any("--negative-prompt is ignored" in str(warning.message) for warning in recwarn)
