import sys
import warnings

import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.ideogram4.cli import ideogram4_generate as cli
from mflux.models.ideogram4.latent_creator.ideogram4_latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


class FakeIdeogram4(FakeModel):
    real = Ideogram4


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeIdeogram4.instances.clear()
    monkeypatch.setattr(cli, "Ideogram4", FakeIdeogram4)
    reset_cli_globals(monkeypatch)
    yield
    FakeIdeogram4.instances.clear()


@pytest.fixture
def lora_file(tmp_path):
    # parse_args validates local LoRA paths; a missing file is exit 2.
    path = tmp_path / "style.safetensors"
    path.write_bytes(b"")
    return path


@pytest.fixture
def full_argv(tmp_path, lora_file):
    # Every value differs from the parser default so a hardcoded default goes red. Sizes are
    # plain ints here. --steps, --guidance and --negative-prompt are left out: this command
    # ignores them, and one test below pins the warnings.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--width", "320", "--height", "256",
        "--preset", "V4_TURBO_12",
        "--strict-caption-validation",
        "--cfg-end", "0.5",
        "--pid-decode", "--pid-degrade-sigma", "0.2",
        "--lora", str(lora_file), "0.5",
        "--no-bake-lora",
        "-q", "8",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ideogram4", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv, model_class=FakeIdeogram4):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ideogram4", *argv])
    cli.main()
    assert len(model_class.instances) == 1
    return model_class.instances[0]


def expected_generate_call(seed):
    # No num_inference_steps and no guidance: the preset owns both.
    return {
        "seed": seed,
        "prompt": "a puffin",
        "width": 320,
        "height": 256,
        "preset": "V4_TURBO_12",
        "strict_caption_validation": True,
        "cfg_end": 0.5,
        "pid_decode": True,
        "pid_degrade_sigma": 0.2,
    }


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, lora_file, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["ideogram-4-fp8"],
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": False,
    }
    assert model.generate_calls == [expected_generate_call(7), expected_generate_call(8)]
    # The fake accepts any size; the real model rejects what its validator rejects.
    for call in model.generate_calls:
        Ideogram4LatentCreator.validate_dimensions(width=call["width"], height=call["height"])
    # With more than one seed parse_args renames --output to <stem>_seed_{seed}; that
    # parse-time rename is part of the contract pinned here.
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_keeps_the_default_flags_off(monkeypatch, tmp_path):
    # full_argv sets every flag to a non-default value, so a flag hardcoded on would pass there.
    model = run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])
    call = model.generate_calls[0]
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["preset"] is None
    assert call["strict_caption_validation"] is False
    assert call["cfg_end"] is None
    assert call["pid_decode"] is False
    assert call["pid_degrade_sigma"] == 0.0
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]
    assert model.init_kwargs["quantize"] is None
    assert model.init_kwargs["bake_lora"] is True


@pytest.mark.fast
def test_main_warns_about_every_ignored_option_before_building_the_model(monkeypatch):
    warnings_at_build = []

    class OrderedFake(FakeIdeogram4):
        def __init__(self, **kwargs):
            warnings_at_build.append(len(warned))
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "Ideogram4", OrderedFake)
    argv = ["--prompt", "x", "--steps", "4", "--guidance", "3", "--negative-prompt", "y"]
    with pytest.warns(UserWarning) as warned:
        model = run_main(monkeypatch, argv, OrderedFake)
    assert sorted(str(w.message).split(";")[0] for w in warned) == [
        "--guidance is ignored",
        "--negative-prompt is ignored",
        "--steps is ignored",
    ]
    assert warnings_at_build == [3]
    assert set(model.generate_calls[0]) == set(expected_generate_call(7))


@pytest.mark.fast
def test_main_passes_a_custom_checkpoint_through(monkeypatch):
    # A repo id is not a registry alias: parse_args moves it to model_path and the command's
    # own entry supplies the config. Nothing is downloaded with the fake.
    # Today's behavior: --base-model does not change the config this command runs.
    model = run_main(monkeypatch, ["--prompt", "x", "--model", "someone/my-finetune", "--base-model", "dev"])
    assert model.init_kwargs["model_path"] == "someone/my-finetune"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["ideogram-4-fp8"]


@pytest.mark.fast
def test_main_rejects_a_foreign_model_before_any_warning_and_builds_nothing(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    # --model is checked first: a bad name fails without the ignored-option warnings.
    monkeypatch.setattr(
        sys, "argv", ["mflux-generate-ideogram4", "--prompt", "x", "--model", "dev", "--negative-prompt", "y"]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            cli.main()
    assert [str(w.message) for w in caught] == []
    assert FakeIdeogram4.instances == []


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, capsys, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_wires_the_stepwise_handler_with_the_ideogram4_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator

    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is Ideogram4LatentCreator


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/12")

    monkeypatch.setattr(FakeIdeogram4, "generate_image", cancel_first)
    model = run_main(monkeypatch, full_argv)
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/12" in out
    assert "Peak MLX memory: " in out
    assert len(model.generate_calls) == 1


@pytest.mark.fast
def test_main_prints_a_missing_prompt_file_and_generates_nothing(monkeypatch, capsys, tmp_path):
    model = run_main(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt")])
    assert "Prompt file does not exist" in capsys.readouterr().out
    assert model.generate_calls == []


@pytest.mark.fast
def test_main_lets_an_unexpected_generate_error_escape_and_still_reports_memory(monkeypatch, capsys, full_argv):
    def boom(self, **kwargs):
        self.bind_generate(**kwargs)
        raise RuntimeError("out of memory")

    monkeypatch.setattr(FakeIdeogram4, "generate_image", boom)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ideogram4", *full_argv])
    with pytest.raises(RuntimeError, match="out of memory"):
        cli.main()
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_reads_the_prompt_file_again_for_each_seed(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a")
    seen = []

    def rewrite_after_first(self, **kwargs):
        self.bind_generate(**kwargs)
        seen.append(kwargs["prompt"])
        prompt_file.write_text("b")
        return FakeModel.generate_image(self, **kwargs)

    monkeypatch.setattr(FakeIdeogram4, "generate_image", rewrite_after_first)
    run_main(
        monkeypatch, ["--prompt-file", str(prompt_file), "--seed", "7", "8", "--output", str(tmp_path / "out.png")]
    )
    assert seen == ["a", "b"]


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_returns_the_fp8_config_for_the_default_alias_without_building(monkeypatch, tmp_path):
    # DEFAULT_MODEL is the alias "ideogram4"; the registry entry behind it is ideogram-4-fp8.
    args = args_for(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt"), "--model", "ideogram4"])
    assert cli.Ideogram4Command.validate(args) is AVAILABLE_MODELS["ideogram-4-fp8"]
    assert FakeIdeogram4.instances == []


@pytest.mark.fast
def test_validate_rejects_a_foreign_model_and_builds_nothing(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.Ideogram4Command.validate(args)
    assert FakeIdeogram4.instances == []


@pytest.mark.fast
def test_load_rejects_a_foreign_model_without_building(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.Ideogram4Command.load(args)
    assert FakeIdeogram4.instances == []


@pytest.mark.fast
def test_load_returns_a_bare_model_with_lora_kwargs(monkeypatch, lora_file):
    model = cli.Ideogram4Command.load(args_for(monkeypatch, ["--prompt", "x", "--lora", str(lora_file), "0.5"]))
    assert model.init_kwargs["lora_paths"] == [str(lora_file)]
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["ideogram-4-fp8"]
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt(monkeypatch, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.Ideogram4Command.load(args)
    image = cli.Ideogram4Command.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7), "prompt": "a different puffin"}]
