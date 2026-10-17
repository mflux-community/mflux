import sys

import pytest

from mflux.cli import check_model
from mflux.models.common.checking.model_checker import ModelChecker, ModelCheckResult


@pytest.mark.fast
def test_main_prints_a_pass_result(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mflux-check", "model", "./model"])
    monkeypatch.setattr(
        ModelChecker,
        "check",
        staticmethod(lambda reference: ModelCheckResult("pass", "safetensors", 13_350_000_000, "qwen-image", ())),
    )

    assert check_model.main() == 0
    assert capsys.readouterr().out == (
        "#################\nMFlux Check Model:\n check: pass\n format: safetensors\n size: 13.35 GB\n"
    )


@pytest.mark.fast
def test_main_returns_failure_for_a_warning(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mflux-check", "model", "--fix", "./model"])
    monkeypatch.setattr(
        ModelChecker,
        "check",
        staticmethod(lambda reference: ModelCheckResult("warn", "mflux", 1, "qwen-image", ("Missing component.",))),
    )
    monkeypatch.setattr(
        ModelChecker,
        "repair",
        staticmethod(lambda reference: "No local replacement is available. The check did not download files."),
    )

    assert check_model.main() == 1
    output = capsys.readouterr().out
    assert "check: warn" in output
    assert "repair: No local replacement is available." in output
