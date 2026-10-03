import sys

import PIL.Image
import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.z_image.cli import z_image_turbo_generate as cli
from mflux.models.z_image.variants.z_image import ZImage
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

# main() warns about --guidance and --negative-prompt (IGNORED_OPTIONS); one test pins the
# warnings, the rest silence them.
pytestmark = pytest.mark.filterwarnings("ignore:--(guidance|negative-prompt) is ignored:UserWarning")


class FakeZImage(FakeModel):
    real = ZImage


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeZImage.instances.clear()
    monkeypatch.setattr(cli, "ZImage", FakeZImage)
    reset_cli_globals(monkeypatch)
    yield
    FakeZImage.instances.clear()


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
        "--width", "2x", "--height", "2x",
        "--image", str(ref_png), "0.55",
        "--guidance", "2.5",
        "--negative-prompt", "blurry",
        "--scheduler", "flow_match_euler_discrete",
        "--pid-decode", "--pid-degrade-sigma", "0.2",
        "--lora", str(lora_file), "0.5",
        "-q", "8",
        "--float32",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-turbo", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-turbo", *argv])
    cli.main()
    assert len(FakeZImage.instances) == 1
    return FakeZImage.instances[0]


def expected_generate_call(seed, ref_png):
    # Literals from the CLI contract (--help and the README), not read back from the code:
    # 2x of a 64x32 reference is 128x64.
    return {
        "seed": seed,
        "prompt": "a puffin",
        "width": 128,
        "height": 64,
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
    with pytest.warns(UserWarning) as warned:
        model = run_main(monkeypatch, full_argv)

    assert sorted(str(w.message).split(";")[0] for w in warned) == [
        "--guidance is ignored",
        "--negative-prompt is ignored",
    ]
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["z-image-turbo"],
        "quantize": 8,
        "model_path": None,
        "float32": True,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": True,
    }
    assert model.generate_calls == [expected_generate_call(7, ref_png), expected_generate_call(8, ref_png)]
    # With more than one seed parse_args renames --output to <stem>_seed_{seed}; that
    # parse-time rename is part of the contract pinned here.
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_load_builds_the_restricted_model_with_lora_kwargs(monkeypatch, lora_file):
    args = args_for(monkeypatch, ["--prompt", "x", "--lora", str(lora_file), "0.5", "-q", "8"])
    model = cli.ZImageTurboCommand.load(args)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["z-image-turbo"],
        "quantize": 8,
        "model_path": None,
        "float32": False,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": True,
    }


@pytest.mark.fast
def test_load_passes_a_custom_checkpoint_through(monkeypatch):
    # A repo id or path is not a registry alias: parse_args moves it to model_path and the
    # family entry supplies the geometry (#694). Nothing is downloaded with the fake.
    args = args_for(monkeypatch, ["--prompt", "x", "--model", "someone/Z-Image-Turbo-4bit"])
    model = cli.ZImageTurboCommand.load(args)
    assert model.init_kwargs["model_path"] == "someone/Z-Image-Turbo-4bit"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["z-image-turbo"]


@pytest.mark.fast
def test_load_rejects_a_base_model_outside_the_turbo_family(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    # --base-model names the family a custom checkpoint belongs to; a foreign one must fail
    # before any weights load instead of quietly running as Turbo.
    args = args_for(monkeypatch, ["--prompt", "x", "--model", "someone/my-finetune", "--base-model", "z-image"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.ZImageTurboCommand.load(args)
    assert FakeZImage.instances == []


@pytest.mark.fast
def test_load_returns_a_model_with_no_callbacks_registered(monkeypatch):
    # MemorySaver.call_before_loop frees the text encoder on a single-seed run, so a model a
    # UI reuses must come back from load() bare.
    args = args_for(monkeypatch, ["--prompt", "x"])
    model = cli.ZImageTurboCommand.load(args)
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt_and_resolved_dims(monkeypatch, ref_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.ZImageTurboCommand.load(args)

    image = cli.ZImageTurboCommand.generate(model, args, 7, "a different puffin")

    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, ref_png), "prompt": "a different puffin"}]


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, capsys, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)

    # register_callbacks adds BatterySaver then MemorySaver; both declare call_before_loop,
    # nothing else does without --stepwise-image-output-dir.
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]
    assert "Peak MLX memory: " in capsys.readouterr().out  # MemorySaver.memory_stats


@pytest.mark.fast
def test_main_wires_the_stepwise_handler_with_the_z_image_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.z_image.latent_creator import ZImageLatentCreator

    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])

    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is ZImageLatentCreator


@pytest.mark.fast
def test_main_keeps_the_default_flags_off(monkeypatch, tmp_path):
    # full_argv sets every flag to a non-default value, so a flag hardcoded on would pass there.
    model = run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])
    assert model.generate_calls[0]["pid_decode"] is False
    assert model.generate_calls[0]["image_path"] is None
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    # The real register_callbacks runs (BatterySaver + MemorySaver); the fake raises what a
    # cancelling in-loop callback would raise on the first seed.
    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeZImage, "generate_image", cancel_first)

    run_main(monkeypatch, full_argv)  # must return, not raise

    model = FakeZImage.instances[0]
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
    assert "Peak MLX memory: " in out
    assert len(model.generate_calls) == 1  # a cancel ends the run; seed 8 never starts


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-turbo", "--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeZImage.instances == []


@pytest.mark.fast
def test_main_prints_a_missing_prompt_file_and_generates_nothing(monkeypatch, capsys, tmp_path):
    # --prompt-file alone parses: the prompt group is required only when the parser is built
    # without supports_metadata_config, and this CLI builds it with it.
    model = run_main(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt")])
    assert "Prompt file does not exist" in capsys.readouterr().out
    assert model.generate_calls == []


@pytest.mark.fast
def test_main_lets_an_unexpected_generate_error_escape_and_still_reports_memory(monkeypatch, capsys, full_argv):
    # Only a cancel or a prompt-file problem is printed and swallowed; anything else must
    # keep its traceback and exit code, and the finally still reports memory.
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-turbo", *full_argv])

    def boom(self, **kwargs):
        self.bind_generate(**kwargs)
        raise RuntimeError("out of memory")

    monkeypatch.setattr(FakeZImage, "generate_image", boom)

    with pytest.raises(RuntimeError, match="out of memory"):
        cli.main()
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_validate_returns_the_turbo_config_without_building_a_model(monkeypatch, tmp_path):
    # A UI checks a request before queuing it: no model, and no input file is opened, so a
    # missing reference image or prompt file is still the generate step's problem.
    args = args_for(
        monkeypatch,
        ["--prompt-file", str(tmp_path / "missing.txt"), "--image", str(tmp_path / "missing.png")],
    )
    assert cli.ZImageTurboCommand.validate(args) == AVAILABLE_MODELS["z-image-turbo"]
    assert FakeZImage.instances == []


@pytest.mark.fast
def test_validate_rejects_a_foreign_model(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.ZImageTurboCommand.validate(args)
    assert FakeZImage.instances == []
