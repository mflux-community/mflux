import sys

import PIL.Image
import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.ernie_image.cli import ernie_image_turbo_generate as cli
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

# main() warns about --negative-prompt (IGNORED_OPTIONS); the tests that pin the warning
# catch it, the rest silence it.
pytestmark = pytest.mark.filterwarnings("ignore:--negative-prompt is ignored:UserWarning")

GUIDANCE_ERROR = "--guidance is only supported for base ERNIE-Image. Use --guidance 1.0 for turbo."


class FakeErnieImage(FakeModel):
    real = ErnieImage


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeErnieImage.instances.clear()
    monkeypatch.setattr(cli, "ErnieImage", FakeErnieImage)
    reset_cli_globals(monkeypatch)
    yield
    FakeErnieImage.instances.clear()


@pytest.fixture
def ref_png(tmp_path):
    path = tmp_path / "ref.png"
    PIL.Image.new("RGB", (64, 32)).save(path)
    return path


@pytest.fixture
def lora_file(tmp_path):
    # parse_args validates local LoRA paths; a missing file is exit 2.
    path = tmp_path / "style.safetensors"
    path.write_bytes(b"")
    return path


@pytest.fixture
def full_argv(tmp_path, ref_png, lora_file):
    # Every value differs from the parser default so a hardcoded default goes red. --guidance
    # is left out on purpose: turbo accepts only 1.0, and an omitted flag must still reach the
    # model as 1.0, never as None.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "2x", "--height", "3x",
        "--image", str(ref_png), "0.55",
        "--negative-prompt", "blurry",
        "--scheduler", "flow_match_euler_discrete",
        "--pid-decode", "--pid-degrade-sigma", "0.2",
        "--lora", str(lora_file), "0.5",
        "--no-bake-lora",
        "-q", "8",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image-turbo", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image-turbo", *argv])
    cli.main()
    assert len(FakeErnieImage.instances) == 1
    return FakeErnieImage.instances[0]


def exit_code_and_stderr(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image-turbo", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return exit_info.value.code, capsys.readouterr().err


def expected_generate_call(seed, ref_png):
    # Literals from the CLI contract, not read back from the code: 2x wide and 3x high on a
    # 64x32 reference is 128x96, and turbo runs at guidance 1.0.
    return {
        "seed": seed,
        "prompt": "a puffin",
        "width": 128,
        "height": 96,
        "guidance": 1.0,
        "image_path": ref_png,
        "num_inference_steps": 3,
        "image_strength": 0.55,
        "scheduler": "flow_match_euler_discrete",
        "negative_prompt": "blurry",
        "pid_decode": True,
        "pid_degrade_sigma": 0.2,
    }


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, ref_png, lora_file, full_argv):
    with pytest.warns(UserWarning) as warned:
        model = run_main(monkeypatch, full_argv)

    assert [str(w.message).split(";")[0] for w in warned] == ["--negative-prompt is ignored"]
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["ernie-image-turbo"],
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": False,
    }
    assert model.generate_calls == [expected_generate_call(7, ref_png), expected_generate_call(8, ref_png)]
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
    assert model.init_kwargs["bake_lora"] is True
    call = model.generate_calls[0]
    assert (call["width"], call["height"]) == (1024, 1024)  # "auto" with no reference image
    assert call["guidance"] == 1.0
    assert call["scheduler"] == "linear"
    assert call["pid_decode"] is False
    assert call["image_path"] is None
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]


@pytest.mark.fast
def test_main_rejects_a_guidance_other_than_one_before_building(monkeypatch, capsys):
    # The boundary values (0, 0.99, 1.01) are pinned at validate(); main() only maps its error.
    code, err = exit_code_and_stderr(monkeypatch, capsys, ["--prompt", "p", "--guidance", "4.0"])
    assert code == 2
    assert GUIDANCE_ERROR in err
    assert FakeErnieImage.instances == []


@pytest.mark.fast
def test_main_warns_about_the_negative_prompt_before_rejecting_guidance(monkeypatch, capsys):
    with pytest.warns(UserWarning, match="--negative-prompt is ignored"):
        code, _ = exit_code_and_stderr(
            monkeypatch, capsys, ["--prompt", "p", "--negative-prompt", "x", "--guidance", "4.0"]
        )
    assert code == 2


@pytest.mark.fast
def test_main_judges_guidance_before_the_model(monkeypatch, capsys):
    # Both are wrong; the guidance message comes first.
    code, err = exit_code_and_stderr(
        monkeypatch, capsys, ["--prompt", "p", "--model", "ernie-image", "--guidance", "4.0"]
    )
    assert code == 2
    assert GUIDANCE_ERROR in err


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    # A --model outside this command's aliases is a traceback, not an exit-2 usage error.
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image-turbo", "--prompt", "p", "--model", "ernie-image"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeErnieImage.instances == []


@pytest.mark.fast
def test_main_todays_behavior_ignores_base_model_for_a_custom_checkpoint(monkeypatch):
    # Today's behavior: this command does not pass --base-model to the registry, so a custom
    # checkpoint always runs on the turbo entry.
    model = run_main(monkeypatch, ["--prompt", "p", "--model", "someone/my-finetune", "--base-model", "ernie-image"])
    assert model.init_kwargs["model_path"] == "someone/my-finetune"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["ernie-image-turbo"]


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, capsys, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)

    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_wires_the_stepwise_handler_with_the_ernie_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.ernie_image.latent_creator import ErnieLatentCreator

    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])

    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is ErnieLatentCreator


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeErnieImage, "generate_image", cancel_first)
    run_main(monkeypatch, full_argv)  # must return, not raise
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
    assert "Peak MLX memory: " in out
    assert len(FakeErnieImage.instances[0].generate_calls) == 1  # seed 8 never starts


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

    monkeypatch.setattr(FakeErnieImage, "generate_image", boom)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image-turbo", *full_argv])
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

    monkeypatch.setattr(FakeErnieImage, "generate_image", rewrite_after_first)
    run_main(
        monkeypatch, ["--prompt-file", str(prompt_file), "--seed", "7", "8", "--output", str(tmp_path / "out.png")]
    )
    assert seen == ["a", "b"]


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_returns_the_turbo_config_without_building_or_reading_files(monkeypatch, tmp_path):
    # A UI checks a request before queuing it: no model, and no input file is opened.
    args = args_for(
        monkeypatch,
        ["--prompt-file", str(tmp_path / "missing.txt"), "--image", str(tmp_path / "missing.png")],
    )
    assert cli.ErnieImageTurboCommand.validate(args) is AVAILABLE_MODELS["ernie-image-turbo"]
    assert FakeErnieImage.instances == []


@pytest.mark.fast
@pytest.mark.parametrize("guidance", ["0", "0.99", "1.01", "4.0"])
def test_validate_rejects_a_guidance_other_than_one(monkeypatch, guidance):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "p", "--guidance", guidance])
    with pytest.raises(ValueError, match="Use --guidance 1.0 for turbo") as raised:
        cli.ErnieImageTurboCommand.validate(args)
    # main() turns this one into exit 2, so it must not be the registry's error type.
    assert not isinstance(raised.value, ModelConfigError)


@pytest.mark.fast
@pytest.mark.parametrize("argv", [["--prompt", "p"], ["--prompt", "p", "--guidance", "1.0"]])
def test_validate_accepts_an_omitted_or_unit_guidance(monkeypatch, argv):
    assert cli.ErnieImageTurboCommand.validate(args_for(monkeypatch, argv)) is AVAILABLE_MODELS["ernie-image-turbo"]


@pytest.mark.fast
def test_validate_rejects_a_foreign_model_and_builds_nothing(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "p", "--model", "ernie-image"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.ErnieImageTurboCommand.validate(args)
    assert FakeErnieImage.instances == []


@pytest.mark.fast
def test_load_rejects_a_guidance_other_than_one_and_builds_nothing(monkeypatch):
    # load() goes through validate(), so a caller that skips validate() builds no model.
    args = args_for(monkeypatch, ["--prompt", "x", "--guidance", "4"])
    with pytest.raises(ValueError, match="Use --guidance 1.0 for turbo"):
        cli.ErnieImageTurboCommand.load(args)
    assert FakeErnieImage.instances == []


@pytest.mark.fast
def test_load_rejects_a_foreign_model_and_builds_nothing(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "ernie-image"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.ErnieImageTurboCommand.load(args)
    assert FakeErnieImage.instances == []


@pytest.mark.fast
def test_load_builds_a_bare_model_with_the_lora_kwargs(monkeypatch, lora_file):
    # MemorySaver frees the text encoder on a single-seed run, so a model a UI reuses must
    # come back from load() with no callbacks registered.
    args = args_for(monkeypatch, ["--prompt", "x", "--lora", str(lora_file), "0.5", "-q", "4"])
    model = cli.ErnieImageTurboCommand.load(args)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["ernie-image-turbo"],
        "quantize": 4,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": True,
    }
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_runs_turbo_at_guidance_one_when_the_flag_was_omitted(monkeypatch):
    # main() does not write the default into args, so a script that parses the flags and
    # calls generate() directly must still run turbo at 1.0, not at None.
    args = args_for(monkeypatch, ["--prompt", "x"])
    model = cli.ErnieImageTurboCommand.load(args)
    cli.ErnieImageTurboCommand.generate(model, args, 7, "x")
    assert model.generate_calls[0]["guidance"] == 1.0
    assert args.guidance is None  # the caller's namespace is left as parsed


@pytest.mark.fast
def test_generate_rejects_a_guidance_other_than_one_and_generates_nothing(monkeypatch):
    # A caller that skips validate() still cannot run turbo at another guidance.
    args = args_for(monkeypatch, ["--prompt", "x", "--guidance", "4"])
    args_ok = args_for(monkeypatch, ["--prompt", "x"])
    model = cli.ErnieImageTurboCommand.load(args_ok)
    with pytest.raises(ValueError, match="Use --guidance 1.0 for turbo"):
        cli.ErnieImageTurboCommand.generate(model, args, 7, "x")
    assert model.generate_calls == []
    assert len(FakeErnieImage.instances) == 1


@pytest.mark.fast
def test_generate_judges_guidance_before_it_reads_the_reference_image(monkeypatch, tmp_path):
    # The default "auto" sizes read the --image header; the guidance rule must win over a bad image.
    args_ok = args_for(monkeypatch, ["--prompt", "x"])
    model = cli.ErnieImageTurboCommand.load(args_ok)
    args = args_for(monkeypatch, ["--prompt", "x", "--guidance", "4", "--image", str(tmp_path / "missing.png")])
    with pytest.raises(ValueError, match="Use --guidance 1.0 for turbo"):
        cli.ErnieImageTurboCommand.generate(model, args, 7, "x")
    assert model.generate_calls == []


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt_and_resolved_dims(monkeypatch, ref_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.ErnieImageTurboCommand.load(args)

    image = cli.ErnieImageTurboCommand.generate(model, args, 7, "a different puffin")

    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, ref_png), "prompt": "a different puffin"}]
