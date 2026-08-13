import sys
import warnings

import pytest

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.flux.cli import flux_generate
from mflux.models.ideogram4.cli import ideogram4_generate
from mflux.models.z_image.cli import z_image_generate, z_image_turbo_generate


class _ModelStubbed(Exception):
    """Raised by the stubbed model constructor: everything before it (parsing and the
    ignored-option warnings) ran through the CLI's real main()."""


def _run_main(monkeypatch, module, model_symbol, argv):
    def boom(*args, **kwargs):
        raise _ModelStubbed

    monkeypatch.setattr(module, model_symbol, boom)
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_ModelStubbed):
            module.main()
    return [str(w.message) for w in caught]


# --- the shared helper ---------------------------------------------------------------


@pytest.mark.fast
def test_provided_option_warns(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x", "--negative-prompt", "blurry"])
    with pytest.warns(UserWarning, match=r"--negative-prompt is ignored; because reasons\."):
        CommandLineParser.warn_ignored_options({"--negative-prompt": "because reasons."})


@pytest.mark.fast
def test_equals_form_warns(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--negative-prompt=blurry"])
    with pytest.warns(UserWarning, match="--negative-prompt is ignored"):
        CommandLineParser.warn_ignored_options({"--negative-prompt": "because reasons."})


@pytest.mark.fast
def test_absent_option_is_silent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CommandLineParser.warn_ignored_options({"--negative-prompt": "because reasons."})


@pytest.mark.fast
def test_each_provided_option_warns_once(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--guidance", "3.5", "--negative-prompt", "blurry"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CommandLineParser.warn_ignored_options(
            {
                "--guidance": "reason a.",
                "--negative-prompt": "reason b.",
                "--steps": "reason c.",
            }
        )
    messages = [str(w.message) for w in caught]
    assert messages == [
        "--guidance is ignored; reason a.",
        "--negative-prompt is ignored; reason b.",
    ]


@pytest.mark.fast
def test_abbreviated_long_options_are_rejected(monkeypatch, capsys):
    # argv-scan detection cannot see argparse prefix matching, so abbreviation is
    # disabled parser-wide: an abbreviated flag must be a hard error, not a silently
    # accepted spelling the warning layer misses.
    parser = CommandLineParser(description="t")
    parser.add_image_generator_arguments()
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x", "--negative", "blurry"])
    with pytest.raises(SystemExit):
        parser.parse_args()
    assert "--negative" in capsys.readouterr().err


# --- the real CLI wirings ------------------------------------------------------------


@pytest.mark.fast
def test_flux_cli_warns_on_negative_prompt(monkeypatch):
    messages = _run_main(
        monkeypatch, flux_generate, "Flux1", ["--model", "schnell", "--prompt", "x", "--negative-prompt", "blurry"]
    )
    assert any("--negative-prompt is ignored" in m for m in messages)


@pytest.mark.fast
def test_ideogram_cli_warns_on_negative_prompt(monkeypatch):
    messages = _run_main(monkeypatch, ideogram4_generate, "Ideogram4", ["--prompt", "x", "--negative-prompt", "y"])
    assert any("--negative-prompt is ignored" in m for m in messages)


@pytest.mark.fast
def test_z_image_turbo_cli_warns_on_both(monkeypatch):
    messages = _run_main(
        monkeypatch,
        z_image_turbo_generate,
        "ZImage",
        ["--prompt", "x", "--guidance", "3.0", "--negative-prompt", "y"],
    )
    assert any("--guidance is ignored" in m for m in messages)
    assert any("--negative-prompt is ignored" in m for m in messages)


# --- the conditional z-image branch ---------------------------------------------------


@pytest.mark.fast
def test_z_image_default_guidance_drops_negative_and_warns(monkeypatch):
    # Omitting --guidance is the common invocation and ALSO disables CFG (the effective
    # default guidance is 0.0), so the negative prompt is dropped and must be warned.
    messages = _run_main(monkeypatch, z_image_generate, "ZImage", ["--prompt", "x", "--negative-prompt", "y"])
    assert any("--negative-prompt has no effect" in m for m in messages)


@pytest.mark.fast
def test_z_image_low_guidance_warns(monkeypatch):
    messages = _run_main(
        monkeypatch, z_image_generate, "ZImage", ["--prompt", "x", "--guidance", "1.0", "--negative-prompt", "y"]
    )
    assert any("--negative-prompt has no effect" in m for m in messages)


@pytest.mark.fast
def test_z_image_high_guidance_is_silent(monkeypatch):
    messages = _run_main(
        monkeypatch, z_image_generate, "ZImage", ["--prompt", "x", "--guidance", "4.0", "--negative-prompt", "y"]
    )
    assert messages == []


@pytest.mark.fast
def test_z_image_cli_resolving_turbo_warns_regardless_of_guidance_value(monkeypatch):
    # --model z-image-turbo through the generic CLI resolves a guidance-distilled
    # config: guidance is forced to 0.0 no matter the flag, so both options must warn
    # even at --guidance 3.0.
    messages = _run_main(
        monkeypatch,
        z_image_generate,
        "ZImage",
        ["--model", "z-image-turbo", "--prompt", "x", "--guidance", "3.0", "--negative-prompt", "y"],
    )
    assert any("--guidance is ignored" in m for m in messages)
    assert any("--negative-prompt is ignored" in m for m in messages)
