import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import PIL.Image
import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.flux2.cli import flux2_edit_generate as cli
from mflux.models.flux2.variants import Flux2KleinEdit
from mflux.utils.exceptions import ModelConfigError
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

GUIDANCE_ERROR = "--guidance is only supported for FLUX.2 base models. Use --guidance 1.0, or declare a base checkpoint with --base-model."  # fmt: off
SMALL_TEXT_ENCODER_ERROR = "--small-text-encoder only works with FLUX.2-klein-4B / klein-base-4B."
NEGATIVE_PROMPT_ERROR = "--negative-prompt is not supported for FLUX.2. Focus on describing what you want."


class FakeFlux2KleinEdit(FakeModel):
    real = Flux2KleinEdit

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # --vae-full-precision switches the decoder's mode on the built model.
        self.vae = SimpleNamespace(decode_mode="legacy")


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeFlux2KleinEdit.instances.clear()
    monkeypatch.setattr(cli, "Flux2KleinEdit", FakeFlux2KleinEdit)
    reset_cli_globals(monkeypatch)
    yield
    FakeFlux2KleinEdit.instances.clear()


@pytest.fixture
def ref_png(tmp_path):
    path = tmp_path / "ref.png"
    PIL.Image.new("RGB", (64, 32)).save(path)
    return path


@pytest.fixture
def tall_png(tmp_path):
    path = tmp_path / "tall.png"
    PIL.Image.new("RGB", (48, 96)).save(path)
    return path


@pytest.fixture
def lora_file(tmp_path):
    # parse_args validates local LoRA paths; a missing file is exit 2.
    path = tmp_path / "style.safetensors"
    path.write_bytes(b"")
    return path


@pytest.fixture
def full_argv(tmp_path, ref_png, tall_png, lora_file):
    # Every value differs from the parser default so a hardcoded default goes red. The base 4B
    # entry takes both a guidance other than 1.0 and the small text encoder.
    return [
        "--prompt", "make it red",
        "--model", "flux2-klein-base-4b",
        "--image-paths", str(ref_png), str(tall_png),
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "2x", "--height", "3x",
        "--guidance", "2.5",
        "--lora", str(lora_file), "0.5",
        "--no-bake-lora",
        "-q", "8",
        "--compute-precision", "float16",
        "--small-text-encoder",
        "--small-decoder",
        "--vae-full-precision",
        "--metadata",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2-edit", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2-edit", *argv])
    cli.main()
    assert len(FakeFlux2KleinEdit.instances) == 1
    return FakeFlux2KleinEdit.instances[0]


def main_exit(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2-edit", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return exit_info.value.code, capsys.readouterr().err


def expected_generate_call(seed, ref_png, tall_png):
    # Literals from the CLI contract: the sizes scale the FIRST image, 2x wide and 3x high on
    # 64x32 is 128x96; every image is passed on, in order.
    return {
        "seed": seed,
        "prompt": "make it red",
        "width": 128,
        "height": 96,
        "guidance": 2.5,
        "image_paths": [ref_png, tall_png],
        "num_inference_steps": 3,
        "scheduler": "flow_match_euler_discrete",
    }


def expected_init_kwargs(lora_file):
    return {
        "model_config": AVAILABLE_MODELS["flux2-klein-base-4b"],
        "quantize": 8,
        "model_path": None,
        "compute_precision": mx.float16,
        "small_text_encoder": True,
        "small_decoder": True,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": False,
    }


# --- main(): the command line ---


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, ref_png, tall_png, lora_file, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == expected_init_kwargs(lora_file)
    assert model.vae.decode_mode == "full_precision"
    assert model.generate_calls == [
        expected_generate_call(7, ref_png, tall_png),
        expected_generate_call(8, ref_png, tall_png),
    ]
    # With more than one seed parse_args renames --output to <stem>_seed_{seed}.
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_runs_the_default_flags(monkeypatch, tmp_path, ref_png):
    # FLUX.2-klein-4B in 4 steps at guidance 1.0, at the first image's own size ("auto").
    argv = ["--prompt", "x", "--image-paths", str(ref_png), "--seed", "3", "--output", str(tmp_path / "o.png")]
    model = run_main(monkeypatch, argv)
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["flux2-klein-4b"]
    assert model.bound_init() | {"model_config": None} == {
        "quantize": None,
        "model_path": None,
        "lora_paths": None,
        "lora_scales": None,
        "bake_lora": True,
        "model_config": None,
        "compute_precision": None,
        "small_text_encoder": False,
        "small_decoder": False,
    }
    assert model.generate_calls == [
        {
            "seed": 3,
            "prompt": "x",
            "width": 64,
            "height": 32,
            "guidance": 1.0,
            "image_paths": [ref_png],
            "num_inference_steps": 4,
            "scheduler": "flow_match_euler_discrete",
        }
    ]
    assert model.images[0].saves == [(str(tmp_path / "o.png"), False)]
    assert model.vae.decode_mode == "legacy"


@pytest.mark.fast
def test_main_sizes_follow_the_first_image_only(monkeypatch, ref_png, tall_png):
    model = run_main(monkeypatch, ["--prompt", "x", "--image-paths", str(tall_png), str(ref_png)])
    assert (model.generate_calls[0]["width"], model.generate_calls[0]["height"]) == (48, 96)
    assert model.generate_calls[0]["image_paths"] == [tall_png, ref_png]


@pytest.mark.fast
def test_main_passes_plain_sizes_through(monkeypatch, ref_png):
    model = run_main(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png), "--width", "320", "--height", "192"])
    assert (model.generate_calls[0]["width"], model.generate_calls[0]["height"]) == (320, 192)


@pytest.mark.fast
def test_main_runs_validate_then_load_then_generate_once_per_seed(monkeypatch, tmp_path, ref_png):
    # The steps a plugin wraps are the ones main() runs: a replaced step is called, in this order.
    calls = []
    real = cli.Flux2EditCommand

    def spy(name):
        step = getattr(real, name)

        def wrapped(*args):
            calls.append(name)
            return step(*args)

        return staticmethod(wrapped)

    for name in ("validate", "load", "generate"):
        monkeypatch.setattr(real, name, spy(name))
    argv = ["--prompt", "x", "--image-paths", str(ref_png), "--seed", "1", "2", "3", "--output", str(tmp_path / "o.png")]  # fmt: skip
    run_main(monkeypatch, argv)
    # load() checks the flags again itself, so a caller of load() alone gets the same refusals.
    assert calls == ["validate", "load", "validate", "generate", "generate", "generate"]


@pytest.mark.fast
def test_main_saves_what_a_wrapped_generate_returns(monkeypatch, tmp_path, ref_png):
    real_generate = cli.Flux2EditCommand.generate
    seen = []

    def wrapped(model, args, seed, prompt):
        seen.append((seed, prompt))
        return real_generate(model, args, seed, prompt.upper())

    monkeypatch.setattr(cli.Flux2EditCommand, "generate", staticmethod(wrapped))
    argv = ["--prompt", "a fox", "--image-paths", str(ref_png), "--seed", "4", "--output", str(tmp_path / "o.png")]
    model = run_main(monkeypatch, argv)
    assert seen == [(4, "a fox")]
    assert model.generate_calls[0]["prompt"] == "A FOX"
    assert model.images[0].saves == [(str(tmp_path / "o.png"), False)]


@pytest.mark.fast
def test_main_refuses_a_negative_prompt_before_resolving_the_model(monkeypatch, capsys, ref_png):
    argv = ["--prompt", "x", "--image-paths", str(ref_png), "--negative-prompt", "y", "--model", "dev"]
    code, err = main_exit(monkeypatch, capsys, argv)
    assert code == 2
    assert NEGATIVE_PROMPT_ERROR in err
    assert FakeFlux2KleinEdit.instances == []


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch, ref_png):
    # A --model outside the family is a traceback, not a usage error, even with a bad guidance too.
    argv = ["mflux-generate-flux2-edit", "--prompt", "x", "--image-paths", str(ref_png), "--model", "dev", "--guidance", "3"]  # fmt: off
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeFlux2KleinEdit.instances == []


@pytest.mark.fast
def test_main_refuses_guidance_on_a_distilled_model_before_the_small_text_encoder(monkeypatch, capsys, ref_png):
    argv = ["--prompt", "x", "--image-paths", str(ref_png), "--model", "flux2-klein-9b", "--guidance", "3", "--small-text-encoder"]  # fmt: off
    code, err = main_exit(monkeypatch, capsys, argv)
    assert code == 2
    assert GUIDANCE_ERROR in err
    assert FakeFlux2KleinEdit.instances == []


@pytest.mark.fast
def test_main_refuses_the_small_text_encoder_on_a_9b_model(monkeypatch, capsys, ref_png):
    argv = ["--prompt", "x", "--image-paths", str(ref_png), "--model", "flux2-klein-9b", "--small-text-encoder"]
    code, err = main_exit(monkeypatch, capsys, argv)
    assert code == 2
    assert SMALL_TEXT_ENCODER_ERROR in err
    assert FakeFlux2KleinEdit.instances == []


@pytest.mark.fast
def test_main_resolves_the_first_image_size_once_for_all_seeds(monkeypatch, ref_png):
    # The size comes from the image header once per run; later seeds do not reopen the file.
    class VanishingReference(FakeFlux2KleinEdit):
        def generate_image(self, **kwargs):
            image = super().generate_image(**kwargs)
            if ref_png.exists():
                ref_png.rename(ref_png.with_suffix(".gone"))
            return image

    monkeypatch.setattr(cli, "Flux2KleinEdit", VanishingReference)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2-edit", "--prompt", "x", "--seed", "1", "2", "--image-paths", str(ref_png)])  # fmt: off
    cli.main()
    model = VanishingReference.instances[0]
    assert [(c["width"], c["height"]) for c in model.generate_calls] == [(64, 32), (64, 32)]


@pytest.mark.fast
def test_main_reports_a_missing_image_before_it_reads_the_prompt(monkeypatch, capsys, tmp_path):
    # A missing first image stops the run before a missing prompt file is noticed; memory is still reported.
    argv = ["mflux-generate-flux2-edit", "--prompt-file", str(tmp_path / "missing.txt"), "--image-paths", str(tmp_path / "x.png")]  # fmt: off
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        cli.main()
    assert FakeFlux2KleinEdit.instances[0].generate_calls == []
    out = capsys.readouterr().out
    assert "Prompt file does not exist" not in out
    assert "Peak MLX memory: " in out


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeFlux2KleinEdit, "generate_image", cancel_first)
    model = run_main(monkeypatch, full_argv)
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
    assert "Peak MLX memory: " in out
    assert len(model.generate_calls) == 1


@pytest.mark.fast
def test_main_prints_a_missing_prompt_file_and_generates_nothing(monkeypatch, capsys, tmp_path, ref_png):
    model = run_main(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt"), "--image-paths", str(ref_png)])
    assert "Prompt file does not exist" in capsys.readouterr().out
    assert model.generate_calls == []


@pytest.mark.fast
def test_main_reads_the_prompt_file_again_for_each_seed(monkeypatch, tmp_path, ref_png):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a")
    seen = []

    def rewrite_after_first(self, **kwargs):
        self.bind_generate(**kwargs)
        seen.append(kwargs["prompt"])
        prompt_file.write_text("b")
        return FakeModel.generate_image(self, **kwargs)

    monkeypatch.setattr(FakeFlux2KleinEdit, "generate_image", rewrite_after_first)
    argv = ["--prompt-file", str(prompt_file), "--image-paths", str(ref_png), "--seed", "7", "8", "--output", str(tmp_path / "o.png")]  # fmt: skip
    run_main(monkeypatch, argv)
    assert seen == ["a", "b"]


@pytest.mark.fast
def test_main_registers_the_callbacks_after_load_and_before_generate(monkeypatch, tmp_path, ref_png):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator

    argv = ["--prompt", "x", "--image-paths", str(ref_png), "--seed", "1", "2", "--low-ram", "--stepwise-image-output-dir", str(tmp_path / "s")]  # fmt: off
    model = run_main(monkeypatch, argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, StepwiseHandler, MemorySaver]
    assert model.before_loop_count_at_generate == [3, 3]
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is Flux2LatentCreator


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv", "key"),
    [
        ([], "flux2-klein-4b"),
        (["--model", "flux2-klein-9b"], "flux2-klein-9b"),
        (["--model", "flux2-klein-base-9b", "--guidance", "4"], "flux2-klein-base-9b"),
        (["--model", "mlx-community/flux2-klein-9b-8bit"], "flux2-klein-9b"),
        (["--model", "/models/my-finetune", "--base-model", "flux2-klein-base-4b"], "flux2-klein-base-4b"),
    ],
)
def test_validate_returns_the_family_config_without_building_or_reading_files(monkeypatch, tmp_path, argv, key):
    args = args_for(
        monkeypatch,
        ["--prompt-file", str(tmp_path / "missing.txt"), "--image-paths", str(tmp_path / "missing.png"), *argv],
    )
    assert cli.Flux2EditCommand.validate(args) is AVAILABLE_MODELS[key]
    assert FakeFlux2KleinEdit.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--model", "flux2-klein-9b", "--guidance", "3"], GUIDANCE_ERROR),
        (["--model", "flux2-klein-4b", "--guidance", "0.5"], GUIDANCE_ERROR),
        (["--model", "mlx-community/flux2-klein-9b-8bit", "--guidance", "3"], GUIDANCE_ERROR),
        (["--model", "/models/my-finetune", "--base-model", "flux2-klein-9b", "--guidance", "3"], GUIDANCE_ERROR),
        (["--model", "flux2-klein-9b", "--small-text-encoder"], SMALL_TEXT_ENCODER_ERROR),
    ],
    ids=["guidance-9b", "guidance-4b", "guidance-inferred", "guidance-declared", "ste-9b"],
)
def test_validate_refuses_what_the_command_refuses(monkeypatch, ref_png, argv, message):
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png), *argv])
    with pytest.raises(ValueError) as exc_info:
        cli.Flux2EditCommand.validate(args)
    assert str(exc_info.value) == message
    assert not isinstance(exc_info.value, ModelConfigError)
    assert FakeFlux2KleinEdit.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "flux2-klein-base-4b", "--guidance", "3", "--small-text-encoder"],
        ["--model", "flux2-klein-9b", "--guidance", "1.0"],
        # A name that matches no family entry says nothing about the weights, so its guidance is not judged.
        ["--model", "/models/my-finetune", "--guidance", "3"],
        ["--small-decoder"],
    ],
    ids=["base-guidance-ste", "distilled-guidance-one", "unjudged-checkpoint", "small-decoder"],
)
def test_validate_accepts_what_the_command_runs(monkeypatch, ref_png, argv):
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png), *argv])
    cli.Flux2EditCommand.validate(args)


@pytest.mark.fast
def test_validate_lets_a_foreign_model_error_escape(monkeypatch, ref_png):
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png), "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.Flux2EditCommand.validate(args)


@pytest.mark.fast
def test_load_builds_the_model_with_the_parsed_options_and_no_callbacks(monkeypatch, lora_file, full_argv):
    model = cli.Flux2EditCommand.load(args_for(monkeypatch, full_argv))
    assert model.init_kwargs == expected_init_kwargs(lora_file)
    assert model.vae.decode_mode == "full_precision"
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_load_passes_a_custom_checkpoint_and_its_inferred_config(monkeypatch, ref_png):
    argv = ["--prompt", "x", "--image-paths", str(ref_png), "--model", "/models/klein-base-9b-tune"]
    args = args_for(monkeypatch, argv)
    model = cli.Flux2EditCommand.load(args)
    assert model.init_kwargs["model_path"] == "/models/klein-base-9b-tune"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["flux2-klein-base-9b"]


@pytest.mark.fast
def test_load_refuses_without_building(monkeypatch, ref_png):
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png), "--model", "flux2-klein-9b", "--small-text-encoder"])  # fmt: off
    with pytest.raises(ValueError, match="only works with FLUX.2-klein-4B"):
        cli.Flux2EditCommand.load(args)
    assert FakeFlux2KleinEdit.instances == []


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt_and_resolved_dims(
    monkeypatch, ref_png, tall_png, full_argv
):
    args = args_for(monkeypatch, full_argv)
    model = cli.Flux2EditCommand.load(args)
    image = cli.Flux2EditCommand.generate(model, args, 7, "make it blue")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, ref_png, tall_png), "prompt": "make it blue"}]


@pytest.mark.fast
def test_generate_without_images_runs_at_the_default_size(monkeypatch, ref_png):
    # A namespace a caller builds can hold no images: there is nothing to size from, so 1024x1024.
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png)])
    args.image_paths = []
    model = cli.Flux2EditCommand.load(args)
    cli.Flux2EditCommand.generate(model, args, 7, "x")
    call = model.generate_calls[0]
    assert call["image_paths"] == []
    assert (call["width"], call["height"]) == (1024, 1024)


@pytest.mark.fast
def test_generate_hands_the_model_paths_when_a_caller_gives_strings(monkeypatch, ref_png, tall_png):
    # A namespace a caller builds can hold plain strings; the model gets Path objects either way.
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png)])
    args.image_paths = [str(tall_png), str(ref_png)]
    model = cli.Flux2EditCommand.load(args)
    cli.Flux2EditCommand.generate(model, args, 7, "x")
    call = model.generate_calls[0]
    assert call["image_paths"] == [Path(str(tall_png)), Path(str(ref_png))]
    assert all(isinstance(p, Path) for p in call["image_paths"])
    assert (call["width"], call["height"]) == (48, 96)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], 1.0),
        (["--model", "flux2-klein-9b"], 1.0),
        (["--model", "flux2-klein-base-4b"], 1.0),
        (["--model", "flux2-klein-base-9b", "--guidance", "4"], 4.0),
    ],
    ids=["distilled-4b-default", "distilled-9b-default", "base-default", "base-explicit"],
)
def test_generate_applies_the_guidance_default_itself(monkeypatch, ref_png, argv, expected):
    # A caller that skips main() gets 1.0, not the parser's None, on distilled and base Klein alike.
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png), *argv])
    model = cli.Flux2EditCommand.load(args)
    cli.Flux2EditCommand.generate(model, args, 7, "x")
    assert model.generate_calls[0]["guidance"] == expected
    if "--guidance" not in argv:
        assert args.guidance is None  # the default is not written back into args


@pytest.mark.fast
def test_generate_always_runs_the_flow_match_scheduler(monkeypatch, ref_png):
    # The shared parser accepts --scheduler; FLUX.2 runs the schedule it was trained with.
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png), "--scheduler", "linear"])
    model = cli.Flux2EditCommand.load(args)
    cli.Flux2EditCommand.generate(model, args, 7, "x")
    assert model.generate_calls[0]["scheduler"] == "flow_match_euler_discrete"


@pytest.mark.fast
def test_the_steps_never_read_the_command_line(monkeypatch, ref_png):
    # The negative-prompt refusal is keyed on sys.argv, so it belongs to main(); a caller of the
    # steps with a stale sys.argv sees neither the refusal nor a warning.
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(ref_png), "--negative-prompt", "y"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cli.Flux2EditCommand.validate(args)
        model = cli.Flux2EditCommand.load(args)
        # A negative_prompt kwarg would fail here: the fake binds every call to the real signature.
        cli.Flux2EditCommand.generate(model, args, 7, "x")
