import sys

import pytest

from mflux.models.boogu.cli import boogu_image_generate as cli
from mflux.models.boogu.variants import BooguImage
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


class FakeBoogu(FakeModel):
    real = BooguImage


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeBoogu.instances.clear()
    monkeypatch.setattr(cli, "BooguImage", FakeBoogu)
    reset_cli_globals(monkeypatch)
    yield
    FakeBoogu.instances.clear()


@pytest.fixture
def full_argv(tmp_path):
    # Every value differs from the parser default so a hardcoded default goes red. Sizes are
    # plain ints here and go to the model unchanged; there are no LoRA flags.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "320", "--height", "192",
        "-q", "8",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-boogu", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv, model_class=FakeBoogu):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-boogu", *argv])
    cli.main()
    assert len(model_class.instances) == 1
    return model_class.instances[0]


def expected_generate_call(seed):
    return {
        "seed": seed,
        "prompt": "a puffin",
        "width": 320,
        "height": 192,
        "num_inference_steps": 3,
    }


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["boogu-image-turbo"],
        "quantize": 8,
        "model_path": None,
    }
    assert model.generate_calls == [expected_generate_call(7), expected_generate_call(8)]
    # With more than one seed parse_args renames --output to <stem>_seed_{seed}; that
    # parse-time rename is part of the contract pinned here.
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_runs_the_default_size_and_steps(monkeypatch, tmp_path):
    # 1024x1024 and Boogu Turbo's 4 steps when no flag says otherwise (the README).
    model = run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])
    call = model.generate_calls[0]
    assert (call["width"], call["height"], call["num_inference_steps"]) == (1024, 1024, 4)
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]
    assert model.init_kwargs["quantize"] is None


@pytest.mark.fast
def test_main_warns_about_every_ignored_option_before_building_the_model(monkeypatch):
    warnings_at_build = []

    class OrderedFake(FakeBoogu):
        def __init__(self, **kwargs):
            warnings_at_build.append(len(warned))
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "BooguImage", OrderedFake)
    with pytest.warns(UserWarning) as warned:
        run_main(monkeypatch, ["--prompt", "x", "--guidance", "3", "--negative-prompt", "y"], OrderedFake)
    assert sorted(str(w.message).split(";")[0] for w in warned) == [
        "--guidance is ignored",
        "--negative-prompt is ignored",
    ]
    assert warnings_at_build == [2]


@pytest.mark.fast
def test_main_passes_a_custom_checkpoint_through(monkeypatch):
    # A repo id is not a registry alias: parse_args moves it to model_path and the command's
    # own entry supplies the config. Nothing is downloaded with the fake.
    # Today's behavior: --base-model does not change the config this command runs.
    model = run_main(monkeypatch, ["--prompt", "x", "--model", "someone/my-finetune", "--base-model", "dev"])
    assert model.init_kwargs["model_path"] == "someone/my-finetune"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["boogu-image-turbo"]


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(sys, "argv", ["mflux-generate-boogu", "--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeBoogu.instances == []


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, capsys, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_hands_the_stepwise_handler_no_latent_creator_today(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler

    # Today's behavior: Boogu has no latent creator, so --stepwise-image-output-dir fails before
    # the first step.
    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is None


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeBoogu, "generate_image", cancel_first)
    model = run_main(monkeypatch, full_argv)
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
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

    monkeypatch.setattr(FakeBoogu, "generate_image", boom)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-boogu", *full_argv])
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

    monkeypatch.setattr(FakeBoogu, "generate_image", rewrite_after_first)
    run_main(
        monkeypatch, ["--prompt-file", str(prompt_file), "--seed", "7", "8", "--output", str(tmp_path / "out.png")]
    )
    assert seen == ["a", "b"]


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_returns_the_boogu_config_without_building_or_reading_files(monkeypatch, tmp_path):
    args = args_for(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt")])
    assert cli.BooguImageCommand.validate(args) is AVAILABLE_MODELS["boogu-image-turbo"]
    assert FakeBoogu.instances == []


@pytest.mark.fast
def test_validate_rejects_a_foreign_model_and_builds_nothing(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.BooguImageCommand.validate(args)
    assert FakeBoogu.instances == []


@pytest.mark.fast
def test_load_rejects_a_foreign_model_without_building(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.BooguImageCommand.load(args)
    assert FakeBoogu.instances == []


@pytest.mark.fast
def test_load_returns_a_bare_model(monkeypatch):
    model = cli.BooguImageCommand.load(args_for(monkeypatch, ["--prompt", "x", "-q", "8"]))
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["boogu-image-turbo"],
        "quantize": 8,
        "model_path": None,
    }
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt(monkeypatch, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.BooguImageCommand.load(args)
    image = cli.BooguImageCommand.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7), "prompt": "a different puffin"}]
