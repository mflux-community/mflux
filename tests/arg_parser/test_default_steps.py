# Every image-generation CLI must default --steps to its own model's step count.
#
# The regression these guard: --steps is resolved in CommandLineParser.parse_args() from
# namespace.model, but most single-model CLIs pick their model in main() (`args.model or
# "..."`), so namespace.model is still None at parse time and every one of them silently
# inherited FLUX.1-dev's 25 steps — 6x the work on a 4-step distillation, half the work
# on a 50-step base model.

import sys

import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.boogu.cli import boogu_image_generate
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.ernie_image.cli import ernie_image_generate, ernie_image_turbo_generate
from mflux.models.fibo.cli import fibo_edit, fibo_generate
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.ideogram4.cli import ideogram4_generate
from mflux.models.krea2.cli import krea2_generate
from mflux.models.lens.cli import lens_generate
from mflux.models.qwen.cli import qwen_image_edit_generate, qwen_image_generate
from mflux.models.z_image.cli import (
    z_image_generate,
    z_image_turbo_generate,
    z_image_turbo_generate_controlnet,
)

# Extra flags a CLI's argparse requires before parse_args() will return.
_EXTRA_ARGV = {
    "flux2_edit_generate": ["--image-paths", "a.png"],
    "qwen_image_edit_generate": ["--image-paths", "a.png"],
    "fibo_edit": ["--image-path", "a.png"],
    "z_image_turbo_generate_controlnet": ["--control", "a.png"],
}

# (module, the model the CLI runs by default, that model's reference step count)
CLI_DEFAULTS = [
    (boogu_image_generate, "boogu-image-turbo", 4),
    (krea2_generate, "krea-2", 8),
    (lens_generate, "lens-turbo", 4),
    (z_image_generate, "z-image", 50),
    (z_image_turbo_generate, "z-image-turbo", 9),
    (z_image_turbo_generate_controlnet, "z-image-turbo-controlnet-union-2.1", 8),
    (flux2_generate, "flux2-klein-4b", 4),
    (flux2_edit_generate, "flux2-klein-4b", 4),
    (qwen_image_generate, "qwen-image", 20),
    (qwen_image_edit_generate, "qwen-image-edit", 20),
    (fibo_generate, "fibo", 50),
    (fibo_edit, "fibo-edit", 50),
    (ernie_image_generate, "ernie-image", 50),
    (ernie_image_turbo_generate, "ernie-image-turbo", 8),
    (ideogram4_generate, "ideogram-4-fp8", 20),
]

# Registry entries with no --steps default of their own. The SeedVR2 upscalers never run
# a step loop, and Krea 2 Raw is the training base — no inference CLI targets it.
NO_DECLARED_STEPS = {"seedvr2-3b", "seedvr2-7b", "krea-2-raw"}


def _parse(monkeypatch, module, extra_argv=()):
    name = module.__name__.rsplit(".", 1)[-1]
    argv = ["prog", "--prompt", "test", *_EXTRA_ARGV.get(name, []), *extra_argv]
    monkeypatch.setattr(sys, "argv", argv)
    return module.build_parser().parse_args()


@pytest.mark.fast
@pytest.mark.parametrize("module, model_key, expected", CLI_DEFAULTS, ids=lambda v: getattr(v, "__name__", v))
def test_default_steps_match_the_cli_model(monkeypatch, module, model_key, expected):
    assert ui_defaults.MODEL_INFERENCE_STEPS[model_key] == expected
    assert _parse(monkeypatch, module).steps == expected


@pytest.mark.fast
@pytest.mark.parametrize("module, model_key, expected", CLI_DEFAULTS, ids=lambda v: getattr(v, "__name__", v))
def test_every_alias_of_the_cli_model_resolves_the_same_steps(monkeypatch, module, model_key, expected):
    # The table used to be keyed by alias, so `--model klein-4b` got 25 while
    # `--model flux2-klein-4b` got 4 — the same checkpoint, two step counts.
    for alias in AVAILABLE_MODELS[model_key].aliases:
        assert _parse(monkeypatch, module, ["--model", alias]).steps == expected, alias


@pytest.mark.fast
@pytest.mark.parametrize("module, model_key, expected", CLI_DEFAULTS, ids=lambda v: getattr(v, "__name__", v))
def test_explicit_steps_always_wins(monkeypatch, module, model_key, expected):
    assert _parse(monkeypatch, module, ["--steps", "3"]).steps == 3


@pytest.mark.fast
@pytest.mark.parametrize("model_key", sorted(set(AVAILABLE_MODELS) - NO_DECLARED_STEPS))
def test_every_registry_model_declares_a_step_count(model_key):
    # Drift guard: adding a model to AVAILABLE_MODELS without a step count silently
    # hands it FLUX.1-dev's 25. Add an entry, or list it in NO_DECLARED_STEPS.
    assert model_key in ui_defaults.MODEL_INFERENCE_STEPS


@pytest.mark.fast
def test_steps_table_is_keyed_by_canonical_registry_keys():
    # Aliases must not be re-added as keys: model_inference_steps() resolves them through
    # the registry, and an alias key here would shadow it and be free to disagree.
    assert set(ui_defaults.MODEL_INFERENCE_STEPS) <= set(AVAILABLE_MODELS)


@pytest.mark.fast
@pytest.mark.parametrize(
    "model_name",
    [None, "some-org/some-finetune", "/local/path/to/checkpoint", "not-a-model"],
)
def test_unknown_models_fall_back_to_the_default(model_name):
    assert ui_defaults.model_inference_steps(model_name) == ui_defaults.DEFAULT_INFERENCE_STEPS


@pytest.mark.fast
def test_repo_id_lookup_prefers_the_lowest_priority_entry():
    # Tongyi-MAI/Z-Image-Turbo is the model_name of both z-image-turbo and its ControlNet.
    # Ties must break the way ConfigResolution's exact-match rule breaks them, or the
    # step count disagrees with the config that actually gets built.
    shared = [k for k, c in AVAILABLE_MODELS.items() if c.model_name == "Tongyi-MAI/Z-Image-Turbo"]
    assert len(shared) > 1
    winner = min(shared, key=lambda k: AVAILABLE_MODELS[k].priority)
    assert ui_defaults.model_inference_steps("Tongyi-MAI/Z-Image-Turbo") == ui_defaults.MODEL_INFERENCE_STEPS[winner]
