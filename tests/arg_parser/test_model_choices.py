# Every name AVAILABLE_MODELS knows must be usable as `--model` / `--base-model`.
#
# The regression these guard: the accepted names were a hand-maintained list in
# ui_defaults (MODEL_CHOICES) that no longer matched the registry. Anything missing from
# it was treated as a local checkpoint directory — `--model lens-turbo` (a canonical key,
# and its own CLI's default model) became model_path='lens-turbo' and died with
# "Model not found" — and `--base-model qwen-image` was rejected by argparse outright.

import sys

import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux.cli import flux_generate
from mflux.models.lens.cli import lens_generate

ALL_NAMES = sorted({*AVAILABLE_MODELS, *(alias for c in AVAILABLE_MODELS.values() for alias in c.aliases)})


def _parse(monkeypatch, module=flux_generate, extra_argv=()):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", *extra_argv])
    return module.build_parser().parse_args()


@pytest.mark.fast
@pytest.mark.parametrize("name", ALL_NAMES)
def test_registry_names_are_never_treated_as_a_local_path(monkeypatch, name):
    # model_path is what PathResolution opens on disk. A built-in name must leave it None
    # so ModelConfig resolves the download instead.
    assert _parse(monkeypatch, extra_argv=["--model", name]).model_path is None


@pytest.mark.fast
@pytest.mark.parametrize("name", ALL_NAMES)
def test_registry_names_resolve_to_a_config(name):
    assert ConfigResolution.resolve(name) is not None


@pytest.mark.fast
def test_model_choices_track_the_registry():
    # Drift guard: the choices are derived, so a new model or alias needs no second edit.
    assert set(ui_defaults.model_choices()) == set(ALL_NAMES)
    assert set(ui_defaults.canonical_model_choices()) == set(AVAILABLE_MODELS)


@pytest.mark.fast
@pytest.mark.parametrize("model_name", ["some-org/some-finetune", "/local/path/to/checkpoint"])
def test_unknown_names_are_still_treated_as_a_path(monkeypatch, model_name):
    assert _parse(monkeypatch, extra_argv=["--model", model_name]).model_path == model_name


@pytest.mark.fast
def test_a_clis_own_default_model_is_accepted_by_name(monkeypatch):
    # The reported repro: `mflux-generate-lens --model lens-turbo` failed on the canonical
    # name of the only model that CLI runs.
    for alias in AVAILABLE_MODELS["lens-turbo"].aliases:
        assert _parse(monkeypatch, lens_generate, ["--model", alias]).model_path is None


@pytest.mark.fast
@pytest.mark.parametrize("base_model", ["qwen-image", "qwen", "klein-9b-kv", "dev", "black-forest-labs/FLUX.1-dev"])
def test_base_model_accepts_every_spelling_the_resolver_accepts(monkeypatch, base_model):
    args = _parse(monkeypatch, extra_argv=["--model", "some-org/some-finetune", "--base-model", base_model])
    assert args.base_model == base_model
    assert ConfigResolution.resolve("some-org/some-finetune", base_model) is not None


@pytest.mark.fast
def test_base_model_still_rejects_an_unknown_name(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, extra_argv=["--model", "some-org/some-finetune", "--base-model", "not-a-model"])


@pytest.mark.fast
def test_base_model_validation_matches_the_resolver():
    # The parser and ConfigResolution.EXPLICIT_BASE must accept exactly the same set,
    # which is why both now read base_model_names().
    allowed = ConfigResolution.base_model_names()
    assert set(ConfigResolution.base_model_keys()) <= set(allowed)
    for key, config in AVAILABLE_MODELS.items():
        if config.base_model is None:
            assert key in allowed, key
            assert all(alias in allowed for alias in config.aliases), key
