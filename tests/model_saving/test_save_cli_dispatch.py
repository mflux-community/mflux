# `mflux-save` must write a checkpoint under the architecture the weights actually have.
#
# The regression these guard: the class was picked by a substring chain over the raw
# `--model` string with `else: Flux1` at the end, so every name it had not been taught was
# saved as a FLUX.1 model — lens, lens-turbo, klein-4b/9b/9b-kv and seedvr2* all did — while
# fibo-edit was saved by the txt2img FIBO class and z-image-controlnet by plain ZImage. None
# of it raised: the mismatch only showed up later as a broken checkpoint.

import contextlib
import functools
import sys
from unittest.mock import patch

import pytest

from mflux.models.common.cli import save
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
from mflux.models.flux.variants.in_context.flux_in_context_dev import Flux1InContextDev
from mflux.models.flux.variants.in_context.flux_in_context_fill import Flux1InContextFill
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image import ZImageTurboControlnet

ALL_NAMES = sorted({*AVAILABLE_MODELS, *(alias for c in AVAILABLE_MODELS.values() for alias in c.aliases)})


def run_save(
    argv_model: str,
    base_model: str | None = None,
    lora_paths: list[str] | None = None,
    lora_flag: str = "--lora",
) -> dict:
    # Runs the CLI with every save-capable constructor stubbed out, so the dispatch is
    # observed without loading a single weight.
    built = {}

    def stub_init(real_init):
        # functools.wraps keeps the real signature visible to inspect.signature, which is
        # what the CLI filters the LoRA kwargs against.
        @functools.wraps(real_init)
        def fake_init(self, *args, **kwargs):
            built["class"] = type(self)
            built["kwargs"] = kwargs

        return fake_init

    argv = ["mflux-save", "--model", argv_model, "--path", "/tmp/mflux-save-test"]
    if base_model:
        argv += ["--base-model", base_model]
    if lora_paths:
        argv += [lora_flag, *lora_paths]

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        for model_class in set(save.MODEL_CLASSES.values()):
            stack.enter_context(patch.object(model_class, "__init__", stub_init(model_class.__init__)))
            stack.enter_context(patch.object(model_class, "save_model", lambda self, base_path: None))
        save.main()

    return built


@pytest.mark.fast
def test_every_registry_entry_is_accounted_for():
    # Drift guard: a model added to AVAILABLE_MODELS must be given a save class or be
    # declared unsaveable. Falling off the end of the table is what produced Flux1.
    assert set(save.MODEL_CLASSES) | set(save.UNSUPPORTED_MODELS) == set(AVAILABLE_MODELS)
    assert not set(save.MODEL_CLASSES) & set(save.UNSUPPORTED_MODELS)


@pytest.mark.fast
@pytest.mark.parametrize("model_class", sorted(set(save.MODEL_CLASSES.values()), key=lambda c: c.__name__))
def test_every_save_class_can_save(model_class):
    assert hasattr(model_class, "save_model")


@pytest.mark.fast
@pytest.mark.parametrize(
    "variant_class",
    [Flux1Depth, Flux1Fill, Flux1Kontext, Flux1InContextFill, Flux1InContextDev, Flux1Redux],
    ids=lambda c: c.__name__,
)
def test_every_flux_variant_class_can_save(variant_class):
    # Conversion tooling (and the Python API generally) drives the generation classes
    # directly; a variant without save_model is the AttributeError of issue #667.
    assert hasattr(variant_class, "save_model")


@pytest.mark.fast
@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_name_dispatches_by_registry_entry(name):
    # Canonical keys and aliases alike go through ConfigResolution, so `klein-4b` and
    # `flux2-klein-4b` cannot disagree the way two substring tests could.
    key = ConfigResolution.resolve_key(name)
    if key in save.UNSUPPORTED_MODELS:
        with pytest.raises(SystemExit) as exit_info:
            run_save(name)
        assert exit_info.value.code == 2
    else:
        assert run_save(name)["class"] is save.MODEL_CLASSES[key]


@pytest.mark.fast
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("klein-4b", Flux2Klein),
        ("flux2-klein-9b-kv", Flux2Klein),
        ("fibo-edit", FIBOEdit),
        ("fiboedit-rmbg", FIBOEdit),
        ("z-image-controlnet", ZImageTurboControlnet),
        # dev-redux must not be saved as plain Flux1: the Redux repo ships no base
        # weights, so the save class is the one that also owns the encoders.
        ("dev-redux", Flux1Redux),
        ("krea-dev", Flux1),  # still Flux.1, and not confused with krea-2
        ("Qwen/Qwen-Image-2512", QwenImage),  # repo ids resolve too, not just aliases
        ("/models/my-qwen-image-finetune", QwenImage),  # inferred from the path name
    ],
)
def test_reported_misclassifications(name, expected):
    assert run_save(name)["class"] is expected


@pytest.mark.fast
@pytest.mark.parametrize("name", ["lens", "lens-turbo", "seedvr2", "seedvr2-7b"])
def test_unsaveable_models_are_rejected_not_saved_as_flux(name):
    with pytest.raises(SystemExit) as exit_info:
        run_save(name)
    assert exit_info.value.code == 2


@pytest.mark.fast
def test_unmatched_name_errors_instead_of_defaulting_to_flux1():
    with pytest.raises(SystemExit) as exit_info:
        run_save("some-org/an-unrecognizable-model")
    assert exit_info.value.code == 2


@pytest.mark.fast
@pytest.mark.parametrize(
    ("base_model", "expected"),
    [
        ("qwen-image", QwenImage),
        ("klein-9b-kv", Flux2Klein),
        ("black-forest-labs/FLUX.1-dev", Flux1),
        # Two entries share the Z-Image-Turbo repo id; addressed by alias, the ControlNet
        # must not swallow the plain one, which a reverse lookup by model_name would do.
        # The bare repo id is a separate, pre-existing resolver ambiguity (both entries
        # carry it as model_name and the ControlNet sorts first by priority) that hits
        # ModelConfig.from_name and every CLI alike, not just this dispatch.
        ("z-image-turbo", save.MODEL_CLASSES["z-image-turbo"]),
        ("z-image-controlnet", ZImageTurboControlnet),
    ],
)
def test_custom_checkpoint_dispatches_on_its_base_model(base_model, expected):
    built = run_save("/models/some-finetune", base_model=base_model)
    assert built["class"] is expected
    assert built["kwargs"]["model_path"] == "/models/some-finetune"


@pytest.mark.fast
def test_hint_lists_the_saveable_models_and_nothing_else():
    # Drift guard on the hint itself: it is derived from MODEL_CLASSES, so an unsaveable
    # model must never appear in it and a saveable one must never be missing.
    listed = {
        name.strip() for name in save.SaveDispatch.saveable_models_hint().split(":", 1)[1].replace("\n", " ").split(",")
    }
    assert listed == set(save.MODEL_CLASSES)
    assert not listed & set(save.UNSUPPORTED_MODELS)


@pytest.mark.fast
@pytest.mark.parametrize("name", ["some-org/an-unrecognizable-model", "lens", "seedvr2-7b"])
def test_every_rejection_names_the_saveable_models(name, capsys):
    # Whichever way the name fails — unresolvable, or resolved but unsaveable — the user
    # is told what would have worked, since --help lists models mflux-save cannot write.
    with pytest.raises(SystemExit):
        run_save(name)
    stderr = capsys.readouterr().err
    assert "mflux-save can save:" in stderr
    assert "qwen-image" in stderr


@pytest.mark.fast
def test_lora_kwargs_match_the_constructor():
    # The Z-Image ControlNet takes LoRA paths but no bake_lora flag: passing it
    # unconditionally is a TypeError before any weight is read.
    assert "bake_lora" not in run_save("z-image-controlnet")["kwargs"]
    assert "lora_paths" in run_save("z-image-controlnet")["kwargs"]
    assert "bake_lora" in run_save("dev")["kwargs"]
    # Boogu and FIBO apply no LoRA at all, so they take no LoRA kwargs to filter.
    assert "lora_paths" not in run_save("boogu")["kwargs"]
    assert "lora_paths" not in run_save("fibo")["kwargs"]


@pytest.mark.fast
@pytest.mark.parametrize("name", ["boogu", "fibo", "fibo-edit"])
# Both spellings, because they reach the check by different routes: --lora-paths lands on
# the namespace directly, while --lora only becomes lora_paths via the atomic-flag
# normalization, so a check reading lora_paths too early would refuse one and not the other.
@pytest.mark.parametrize("lora_flag", ["--lora", "--lora-paths"])
def test_lora_is_refused_for_models_that_cannot_apply_it(name, lora_flag, tmp_path, capsys):
    # The signature filter silently dropped --lora for these, so mflux-save wrote an
    # unmodified checkpoint while the user believed the adapter had been merged in.
    lora_file = tmp_path / "adapter.safetensors"
    lora_file.touch()
    with pytest.raises(SystemExit) as exit_info:
        run_save(name, lora_paths=[str(lora_file)], lora_flag=lora_flag)
    assert exit_info.value.code == 2
    assert "cannot apply LoRA weights" in capsys.readouterr().err
