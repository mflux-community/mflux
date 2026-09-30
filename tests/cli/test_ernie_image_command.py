import sys

import PIL.Image
import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.ernie_image.cli import ernie_image_generate as cli
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


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
    # Every value differs from the parser default so a hardcoded default goes red.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "2x", "--height", "3x",
        "--image", str(ref_png), "0.55",
        "--guidance", "2.5",
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
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image", *argv])
    cli.main()
    assert len(FakeErnieImage.instances) == 1
    return FakeErnieImage.instances[0]


def expected_generate_call(seed, ref_png):
    # 2x wide and 3x high on a 64x32 reference is 128x96.
    return {
        "seed": seed,
        "prompt": "a puffin",
        "width": 128,
        "height": 96,
        "guidance": 2.5,
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
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["ernie-image"],
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
def test_main_runs_the_parser_defaults(monkeypatch, tmp_path):
    model = run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])
    call = model.generate_calls[0]
    assert call["guidance"] == 4.0
    assert call["scheduler"] == "linear"
    assert (call["width"], call["height"]) == (1024, 1024)  # "auto" with no reference image
    assert call["pid_decode"] is False
    assert model.init_kwargs["bake_lora"] is True
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]


@pytest.mark.fast
def test_main_todays_behavior_runs_linear_for_the_equals_sign_scheduler(monkeypatch):
    # Today's behavior: the command sets linear whenever the bare token --scheduler is missing
    # from argv, so the --scheduler=NAME spelling runs linear.
    model = run_main(monkeypatch, ["--prompt", "x", "--scheduler=flow_match_euler_discrete"])
    assert model.generate_calls[0]["scheduler"] == "linear"


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image", "--prompt", "p", "--model", "ernie-image-turbo"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeErnieImage.instances == []


@pytest.mark.fast
def test_main_todays_behavior_ignores_base_model_for_a_custom_checkpoint(monkeypatch):
    # Today's behavior: this command does not pass --base-model to the registry, so a custom
    # checkpoint always runs on the base entry.
    model = run_main(
        monkeypatch, ["--prompt", "p", "--model", "someone/my-finetune", "--base-model", "ernie-image-turbo"]
    )
    assert model.init_kwargs["model_path"] == "someone/my-finetune"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["ernie-image"]


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_with_the_ernie_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.ernie_image.latent_creator import ErnieLatentCreator

    model = run_main(
        monkeypatch, ["--prompt", "x", "--seed", "7", "8", "--stepwise-image-output-dir", str(tmp_path / "steps")]
    )

    # register_callbacks adds the battery saver, the stepwise handler, then the memory saver.
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, StepwiseHandler, MemorySaver]
    assert model.before_loop_count_at_generate == [3, 3]
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert [h.latent_creator for h in handlers] == [ErnieLatentCreator]


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeErnieImage, "generate_image", cancel_first)
    run_main(monkeypatch, full_argv)
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
    assert "Peak MLX memory: " in out
    assert len(FakeErnieImage.instances[0].generate_calls) == 1


@pytest.mark.fast
def test_main_lets_an_unexpected_generate_error_escape_and_still_reports_memory(monkeypatch, capsys, full_argv):
    def boom(self, **kwargs):
        self.bind_generate(**kwargs)
        raise RuntimeError("out of memory")

    monkeypatch.setattr(FakeErnieImage, "generate_image", boom)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-ernie-image", *full_argv])
    with pytest.raises(RuntimeError, match="out of memory"):
        cli.main()
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_prints_a_missing_prompt_file_and_generates_nothing(monkeypatch, capsys, tmp_path):
    model = run_main(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt")])
    assert "Prompt file does not exist" in capsys.readouterr().out
    assert model.generate_calls == []


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
def test_validate_returns_the_base_config_without_building_or_reading_files(monkeypatch, tmp_path):
    args = args_for(
        monkeypatch,
        ["--prompt-file", str(tmp_path / "missing.txt"), "--image", str(tmp_path / "missing.png")],
    )
    assert cli.ErnieImageCommand.validate(args) is AVAILABLE_MODELS["ernie-image"]
    assert FakeErnieImage.instances == []


@pytest.mark.fast
def test_validate_rejects_a_foreign_model_and_builds_nothing(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "p", "--model", "ernie-image-turbo"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.ErnieImageCommand.validate(args)
    assert FakeErnieImage.instances == []


@pytest.mark.fast
def test_load_builds_a_bare_model_with_the_lora_kwargs(monkeypatch, lora_file):
    args = args_for(monkeypatch, ["--prompt", "x", "--lora", str(lora_file), "0.5", "-q", "4"])
    model = cli.ErnieImageCommand.load(args)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["ernie-image"],
        "quantize": 4,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": True,
    }
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_runs_guidance_four_when_the_namespace_holds_none(monkeypatch):
    # The parser defaults --guidance to 4.0 here, but a caller that builds or edits its own
    # namespace can hand over None, and generate() must then run the command's default, 4.0.
    args = args_for(monkeypatch, ["--prompt", "x"])
    args.guidance = None
    model = cli.ErnieImageCommand.load(args)
    cli.ErnieImageCommand.generate(model, args, 7, "x")
    assert model.generate_calls[0]["guidance"] == 4.0
    assert args.guidance is None


@pytest.mark.fast
def test_generate_uses_the_scheduler_in_args_as_given(monkeypatch):
    # argv here has no --scheduler token; generate() must not apply the command line's
    # sys.argv default, only what the caller put in args.
    args = args_for(monkeypatch, ["--prompt", "x"])
    args.scheduler = "flow_match_euler_discrete"
    model = cli.ErnieImageCommand.load(args)
    cli.ErnieImageCommand.generate(model, args, 7, "x")
    assert model.generate_calls[0]["scheduler"] == "flow_match_euler_discrete"


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt_and_resolved_dims(monkeypatch, ref_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.ErnieImageCommand.load(args)

    image = cli.ErnieImageCommand.generate(model, args, 7, "a different puffin")

    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, ref_png), "prompt": "a different puffin"}]
