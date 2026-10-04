import sys
import warnings

import PIL.Image
import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.krea2.cli import krea2_generate as cli
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


class FakeKrea2(FakeModel):
    real = Krea2


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeKrea2.instances.clear()
    monkeypatch.setattr(cli, "Krea2", FakeKrea2)
    reset_cli_globals(monkeypatch)
    yield
    FakeKrea2.instances.clear()


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
    # Every value differs from the parser default so a hardcoded default goes red. Guidance is
    # not 1.0, so main() has nothing to warn about.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "2x", "--height", "3x",
        "--image", str(ref_png), "0.55",
        "--guidance", "2.5",
        "--negative-prompt", "blurry",
        "--scheduler", "euler",
        "--pid-decode", "--pid-degrade-sigma", "0.2",
        "--lora", str(lora_file), "0.5",
        "--no-bake-lora",
        "-q", "8",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-krea2", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv, model_class=FakeKrea2):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-krea2", *argv])
    cli.main()
    assert len(model_class.instances) == 1
    return model_class.instances[0]


def expected_generate_call(seed, ref_png):
    # Literals from the CLI contract, not read back from the code: 2x wide and 3x high on a
    # 64x32 reference is 128x96.
    return {
        "seed": seed,
        "prompt": "a puffin",
        "num_inference_steps": 3,
        "height": 96,
        "width": 128,
        "guidance": 2.5,
        "scheduler": "euler",
        "negative_prompt": "blurry",
        "image_path": ref_png,
        "image_strength": 0.55,
        "pid_decode": True,
        "pid_degrade_sigma": 0.2,
    }


def negative_prompt_warnings(caught):
    return [str(w.message) for w in caught if "--negative-prompt" in str(w.message)]


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, ref_png, lora_file, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["krea-2"],
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
def test_main_runs_the_distilled_default_guidance_and_the_default_flags(monkeypatch, tmp_path):
    # Krea-2 Turbo's CFG default is 1.0 (the README), not the parser's None; sizes default to
    # 1024x1024 when there is no reference image.
    model = run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])
    call = model.generate_calls[0]
    assert call["guidance"] == 1.0
    assert (call["width"], call["height"]) == (1024, 1024)
    assert call["image_path"] is None
    assert call["pid_decode"] is False
    assert call["num_inference_steps"] == 8
    assert call["scheduler"] == "linear"
    assert call["pid_degrade_sigma"] == 0.0
    assert call["image_strength"] == ui_defaults.IMAGE_STRENGTH
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]
    assert model.init_kwargs["quantize"] is None
    assert model.init_kwargs["bake_lora"] is True


@pytest.mark.fast
@pytest.mark.parametrize("guidance_argv", [[], ["--guidance", "1.0"]], ids=["default", "explicit-1.0"])
def test_main_warns_about_a_negative_prompt_at_guidance_one_once_the_model_is_built(monkeypatch, guidance_argv):
    warnings_at_build = []
    warnings_at_generate = []

    class OrderedFake(FakeKrea2):
        def __init__(self, **kwargs):
            warnings_at_build.append(len(caught))
            super().__init__(**kwargs)

        def generate_image(self, **kwargs):
            warnings_at_generate.append(len(negative_prompt_warnings(caught)))
            return super().generate_image(**kwargs)

    monkeypatch.setattr(cli, "Krea2", OrderedFake)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = run_main(
            monkeypatch,
            ["--prompt", "x", "--negative-prompt", "y", "--seed", "1", "2", *guidance_argv],
            OrderedFake,
        )
    # The message is a literal, so a changed or dropped reason shows up here.
    assert negative_prompt_warnings(caught) == [
        "--negative-prompt is ignored at guidance 1.0; the encoder builds the unconditional branch only when "
        "guidance != 1.0, so at the distilled default of 1.0 the negative prompt is never encoded."
    ]
    # The warning fires once, after the weights load and before the first image.
    assert warnings_at_build == [0]
    assert warnings_at_generate == [1, 1]
    assert model.generate_calls[0]["negative_prompt"] == "y"


@pytest.mark.fast
def test_main_warns_about_a_negative_prompt_before_it_reads_the_prompt(monkeypatch, tmp_path):
    # A missing prompt file ends the run; the warning has already fired by then.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_main(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt"), "--negative-prompt", "y"])
    assert len(negative_prompt_warnings(caught)) == 1


@pytest.mark.fast
@pytest.mark.parametrize("guidance", ["0", "0.5", "1.5"])
def test_main_is_silent_about_a_negative_prompt_at_other_guidance(monkeypatch, guidance):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = run_main(monkeypatch, ["--prompt", "x", "--negative-prompt", "y", "--guidance", guidance])
    assert negative_prompt_warnings(caught) == []
    assert model.generate_calls[0]["guidance"] == float(guidance)


@pytest.mark.fast
def test_main_is_silent_without_a_negative_prompt(monkeypatch):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_main(monkeypatch, ["--prompt", "x"])
    assert negative_prompt_warnings(caught) == []


@pytest.mark.fast
def test_main_passes_a_custom_checkpoint_through(monkeypatch):
    # A repo id is not a registry alias: parse_args moves it to model_path and the command's
    # own entry supplies the config. Nothing is downloaded with the fake.
    # Today's behavior: --base-model does not change the config this command runs.
    model = run_main(monkeypatch, ["--prompt", "x", "--model", "someone/my-finetune", "--base-model", "krea-2-raw"])
    assert model.init_kwargs["model_path"] == "someone/my-finetune"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["krea-2"]


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(sys, "argv", ["mflux-generate-krea2", "--prompt", "x", "--model", "krea-2-raw"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeKrea2.instances == []


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, capsys, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_wires_the_stepwise_handler_with_the_krea2_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.krea2.latent_creator import Krea2LatentCreator

    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is Krea2LatentCreator


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeKrea2, "generate_image", cancel_first)
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

    monkeypatch.setattr(FakeKrea2, "generate_image", boom)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-krea2", *full_argv])
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

    monkeypatch.setattr(FakeKrea2, "generate_image", rewrite_after_first)
    run_main(
        monkeypatch, ["--prompt-file", str(prompt_file), "--seed", "7", "8", "--output", str(tmp_path / "out.png")]
    )
    assert seen == ["a", "b"]


@pytest.mark.fast
def test_main_reports_a_missing_reference_image_before_it_reads_the_prompt(monkeypatch, tmp_path):
    # Sizes left at "auto" follow the reference image, and main() resolves them before the
    # seed loop, so a missing image stops the run before a missing prompt file is noticed.
    with pytest.raises(FileNotFoundError):
        run_main(
            monkeypatch,
            ["--prompt-file", str(tmp_path / "missing.txt"), "--image", str(tmp_path / "missing.png")],
        )
    assert FakeKrea2.instances[0].generate_calls == []


@pytest.mark.fast
def test_main_reads_the_reference_image_size_once_for_all_seeds(monkeypatch, ref_png):
    # The size comes from the image header once per run; later seeds do not reopen the file.
    class VanishingReference(FakeKrea2):
        def generate_image(self, **kwargs):
            image = super().generate_image(**kwargs)
            if ref_png.exists():
                ref_png.rename(ref_png.with_suffix(".gone"))
            return image

    monkeypatch.setattr(cli, "Krea2", VanishingReference)
    model = run_main(monkeypatch, ["--prompt", "x", "--seed", "1", "2", "--image", str(ref_png)], VanishingReference)
    assert [(c["width"], c["height"]) for c in model.generate_calls] == [(64, 32), (64, 32)]


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_returns_the_krea2_config_without_building_or_reading_files(monkeypatch, tmp_path):
    args = args_for(
        monkeypatch,
        ["--prompt-file", str(tmp_path / "missing.txt"), "--image", str(tmp_path / "missing.png")],
    )
    assert cli.Krea2Command.validate(args) is AVAILABLE_MODELS["krea-2"]
    assert FakeKrea2.instances == []


@pytest.mark.fast
def test_validate_rejects_a_foreign_model_and_builds_nothing(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "krea-2-raw"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.Krea2Command.validate(args)
    assert FakeKrea2.instances == []


@pytest.mark.fast
def test_load_rejects_a_foreign_model_without_building(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "krea-2-raw"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.Krea2Command.load(args)
    assert FakeKrea2.instances == []


@pytest.mark.fast
def test_load_returns_a_bare_model_with_lora_kwargs(monkeypatch, lora_file):
    model = cli.Krea2Command.load(args_for(monkeypatch, ["--prompt", "x", "--lora", str(lora_file), "0.5"]))
    assert model.init_kwargs["lora_paths"] == [str(lora_file)]
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["krea-2"]
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt_and_resolved_dims(monkeypatch, ref_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.Krea2Command.load(args)
    image = cli.Krea2Command.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, ref_png), "prompt": "a different puffin"}]


@pytest.mark.fast
def test_generate_applies_the_default_guidance_itself(monkeypatch):
    # A UI that skips main() must still get Krea-2's 1.0, not the parser's None.
    args = args_for(monkeypatch, ["--prompt", "x"])
    model = cli.Krea2Command.load(args)
    cli.Krea2Command.generate(model, args, 7, "x")
    assert model.generate_calls[0]["guidance"] == 1.0
    assert args.guidance is None  # the default is not written back into args


@pytest.mark.fast
def test_the_steps_never_warn_even_when_the_command_line_would(monkeypatch):
    # The negative-prompt warning reads sys.argv, so it belongs to main(); a UI calling the
    # steps with a stale sys.argv must not see it.
    args = args_for(monkeypatch, ["--prompt", "x", "--negative-prompt", "y"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cli.Krea2Command.validate(args)
        model = cli.Krea2Command.load(args)
        cli.Krea2Command.generate(model, args, 7, "x")
