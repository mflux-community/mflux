import sys
import warnings
from types import SimpleNamespace

import mlx.core as mx
import PIL.Image
import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.flux2.cli import flux2_generate as cli
from mflux.models.flux2.variants import Flux2Klein
from mflux.utils.exceptions import ModelConfigError
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

GUIDANCE_ERROR = "--guidance is only supported for FLUX.2 base models. Use --guidance 1.0, or declare a base checkpoint with --base-model."  # fmt: off
SMALL_TEXT_ENCODER_ERROR = "--small-text-encoder only works with FLUX.2-klein-4B / klein-base-4B."
SMALL_DECODER_ERROR = "--small-decoder cannot be used with --pid-decode: PiD replaces the VAE decoder."
NEGATIVE_PROMPT_ERROR = "--negative-prompt is not supported for FLUX.2. Focus on describing what you want."


class FakeFlux2Klein(FakeModel):
    real = Flux2Klein

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # --vae-full-precision switches the decoder's mode on the built model.
        self.vae = SimpleNamespace(decode_mode="legacy")


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeFlux2Klein.instances.clear()
    monkeypatch.setattr(cli, "Flux2Klein", FakeFlux2Klein)
    reset_cli_globals(monkeypatch)
    yield
    FakeFlux2Klein.instances.clear()


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
    # Every value differs from the parser default so a hardcoded default goes red. The base 4B
    # entry takes both a guidance other than 1.0 and the small text encoder.
    return [
        "--prompt", "a puffin",
        "--model", "flux2-klein-base-4b",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "2x", "--height", "3x",
        "--image", str(ref_png), "0.55",
        "--guidance", "2.5",
        "--pid-decode", "--pid-degrade-sigma", "0.2",
        "--lora", str(lora_file), "0.5",
        "--no-bake-lora",
        "-q", "8",
        "--compute-precision", "float16",
        "--small-text-encoder",
        "--vae-full-precision",
        "--metadata",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2", *argv])
    cli.main()
    assert len(FakeFlux2Klein.instances) == 1
    return FakeFlux2Klein.instances[0]


def main_exit(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return exit_info.value.code, capsys.readouterr().err


def expected_generate_call(seed, ref_png):
    # Literals from the CLI contract: 2x wide and 3x high on a 64x32 reference is 128x96.
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
        "pid_decode": True,
        "pid_degrade_sigma": 0.2,
    }


def expected_init_kwargs(lora_file):
    return {
        "model_config": AVAILABLE_MODELS["flux2-klein-base-4b"],
        "quantize": 8,
        "model_path": None,
        "compute_precision": mx.float16,
        "small_text_encoder": True,
        "small_decoder": False,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": False,
    }


# --- main(): the command line ---


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, ref_png, lora_file, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == expected_init_kwargs(lora_file)
    assert model.vae.decode_mode == "full_precision"
    assert model.generate_calls == [expected_generate_call(7, ref_png), expected_generate_call(8, ref_png)]
    # With more than one seed parse_args renames --output to <stem>_seed_{seed}.
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_runs_the_default_flags(monkeypatch, tmp_path):
    # FLUX.2-klein-4B in 4 steps at guidance 1.0 and 1024x1024 when there is no reference image.
    model = run_main(monkeypatch, ["--prompt", "x", "--seed", "3", "--output", str(tmp_path / "out.png")])
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
            "width": 1024,
            "height": 1024,
            "guidance": 1.0,
            "image_path": None,
            "num_inference_steps": 4,
            "image_strength": 0.4,
            "scheduler": "flow_match_euler_discrete",
            "pid_decode": False,
            "pid_degrade_sigma": 0.0,
        }
    ]
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]
    assert model.vae.decode_mode == "legacy"


@pytest.mark.fast
def test_main_runs_validate_then_load_then_generate_once_per_seed(monkeypatch, tmp_path):
    # The steps a plugin wraps are the ones main() runs: a replaced step is called, in this order.
    calls = []
    real = cli.Flux2Command

    def spy(name):
        step = getattr(real, name)

        def wrapped(*args):
            calls.append(name)
            return step(*args)

        return staticmethod(wrapped)

    for name in ("validate", "load", "generate"):
        monkeypatch.setattr(real, name, spy(name))
    run_main(monkeypatch, ["--prompt", "x", "--seed", "1", "2", "3", "--output", str(tmp_path / "out.png")])
    # load() checks the flags again itself, so a caller of load() alone gets the same refusals.
    assert calls == ["validate", "load", "validate", "generate", "generate", "generate"]


@pytest.mark.fast
def test_main_saves_what_a_wrapped_generate_returns(monkeypatch, tmp_path):
    real_generate = cli.Flux2Command.generate
    seen = []

    def wrapped(model, args, seed, prompt):
        seen.append((seed, prompt))
        return real_generate(model, args, seed, prompt.upper())

    monkeypatch.setattr(cli.Flux2Command, "generate", staticmethod(wrapped))
    model = run_main(monkeypatch, ["--prompt", "a fox", "--seed", "4", "--output", str(tmp_path / "out.png")])
    assert seen == [(4, "a fox")]
    assert model.generate_calls[0]["prompt"] == "A FOX"
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]


@pytest.mark.fast
def test_main_refuses_a_negative_prompt_before_anything_else(monkeypatch, capsys):
    # Checked first, so a run also wrong on --small-decoder reports the negative prompt.
    code, err = main_exit(
        monkeypatch, capsys, ["--prompt", "x", "--negative-prompt", "y", "--small-decoder", "--pid-decode"]
    )
    assert code == 2
    assert NEGATIVE_PROMPT_ERROR in err
    assert FakeFlux2Klein.instances == []


@pytest.mark.fast
def test_main_refuses_the_small_decoder_with_pid_before_resolving_the_model(monkeypatch, capsys):
    code, err = main_exit(monkeypatch, capsys, ["--prompt", "x", "--small-decoder", "--pid-decode", "--model", "dev"])
    assert code == 2
    assert SMALL_DECODER_ERROR in err
    assert FakeFlux2Klein.instances == []


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch):
    # A --model outside the family is a traceback, not a usage error, even with a bad guidance too.
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2", "--prompt", "x", "--model", "dev", "--guidance", "3"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeFlux2Klein.instances == []


@pytest.mark.fast
def test_main_refuses_guidance_on_a_distilled_model_before_the_small_text_encoder(monkeypatch, capsys):
    argv = ["--prompt", "x", "--model", "flux2-klein-9b", "--guidance", "3", "--small-text-encoder"]
    code, err = main_exit(monkeypatch, capsys, argv)
    assert code == 2
    assert GUIDANCE_ERROR in err
    assert FakeFlux2Klein.instances == []


@pytest.mark.fast
def test_main_refuses_the_small_text_encoder_on_a_9b_model(monkeypatch, capsys):
    code, err = main_exit(monkeypatch, capsys, ["--prompt", "x", "--model", "flux2-klein-9b", "--small-text-encoder"])
    assert code == 2
    assert SMALL_TEXT_ENCODER_ERROR in err
    assert FakeFlux2Klein.instances == []


@pytest.mark.fast
def test_main_resolves_the_reference_image_size_once_for_all_seeds(monkeypatch, ref_png):
    # The size comes from the image header once per run; later seeds do not reopen the file.
    class VanishingReference(FakeFlux2Klein):
        def generate_image(self, **kwargs):
            image = super().generate_image(**kwargs)
            if ref_png.exists():
                ref_png.rename(ref_png.with_suffix(".gone"))
            return image

    monkeypatch.setattr(cli, "Flux2Klein", VanishingReference)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2", "--prompt", "x", "--seed", "1", "2", "--image", str(ref_png)])  # fmt: off
    cli.main()
    model = VanishingReference.instances[0]
    assert [(c["width"], c["height"]) for c in model.generate_calls] == [(64, 32), (64, 32)]


@pytest.mark.fast
def test_main_reports_a_missing_reference_image_before_it_reads_the_prompt(monkeypatch, capsys, tmp_path):
    # A missing image stops the run before a missing prompt file is noticed; memory is still reported.
    monkeypatch.setattr(
        sys,
        "argv",
        ["mflux-generate-flux2", "--prompt-file", str(tmp_path / "missing.txt"), "--image", str(tmp_path / "x.png")],
    )
    with pytest.raises(FileNotFoundError):
        cli.main()
    assert FakeFlux2Klein.instances[0].generate_calls == []
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

    monkeypatch.setattr(FakeFlux2Klein, "generate_image", cancel_first)
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
def test_main_reads_the_prompt_file_again_for_each_seed(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a")
    seen = []

    def rewrite_after_first(self, **kwargs):
        self.bind_generate(**kwargs)
        seen.append(kwargs["prompt"])
        prompt_file.write_text("b")
        return FakeModel.generate_image(self, **kwargs)

    monkeypatch.setattr(FakeFlux2Klein, "generate_image", rewrite_after_first)
    run_main(
        monkeypatch, ["--prompt-file", str(prompt_file), "--seed", "7", "8", "--output", str(tmp_path / "out.png")]
    )
    assert seen == ["a", "b"]


@pytest.mark.fast
def test_main_registers_the_callbacks_after_load_and_before_generate(monkeypatch, tmp_path):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator

    argv = ["--prompt", "x", "--seed", "1", "2", "--low-ram", "--stepwise-image-output-dir", str(tmp_path / "s")]
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
def test_validate_returns_the_family_config_without_building(monkeypatch, tmp_path, argv, key):
    args = args_for(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt"), *argv])
    assert cli.Flux2Command.validate(args) is AVAILABLE_MODELS[key]
    assert FakeFlux2Klein.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--small-decoder", "--pid-decode"], SMALL_DECODER_ERROR),
        (["--model", "flux2-klein-9b", "--guidance", "3"], GUIDANCE_ERROR),
        (["--model", "flux2-klein-4b", "--guidance", "0.5"], GUIDANCE_ERROR),
        (["--model", "mlx-community/flux2-klein-9b-8bit", "--guidance", "3"], GUIDANCE_ERROR),
        (["--model", "/models/my-finetune", "--base-model", "flux2-klein-9b", "--guidance", "3"], GUIDANCE_ERROR),
        (["--model", "flux2-klein-9b", "--small-text-encoder"], SMALL_TEXT_ENCODER_ERROR),
    ],
    ids=["small-decoder-pid", "guidance-9b", "guidance-4b", "guidance-inferred", "guidance-declared", "ste-9b"],
)
def test_validate_refuses_what_the_command_refuses(monkeypatch, argv, message):
    args = args_for(monkeypatch, ["--prompt", "x", *argv])
    with pytest.raises(ValueError) as exc_info:
        cli.Flux2Command.validate(args)
    assert str(exc_info.value) == message
    assert not isinstance(exc_info.value, ModelConfigError)
    assert FakeFlux2Klein.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "flux2-klein-base-4b", "--guidance", "3", "--small-text-encoder"],
        ["--model", "flux2-klein-9b", "--guidance", "1.0"],
        # A name that matches no family entry says nothing about the weights, so its guidance is not judged.
        ["--model", "/models/my-finetune", "--guidance", "3"],
        ["--small-decoder"],
        ["--pid-decode"],
    ],
    ids=["base-guidance-ste", "distilled-guidance-one", "unjudged-checkpoint", "small-decoder", "pid"],
)
def test_validate_accepts_what_the_command_runs(monkeypatch, argv):
    args = args_for(monkeypatch, ["--prompt", "x", *argv])
    cli.Flux2Command.validate(args)


@pytest.mark.fast
def test_validate_lets_a_foreign_model_error_escape(monkeypatch):
    args = args_for(monkeypatch, ["--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.Flux2Command.validate(args)


@pytest.mark.fast
def test_load_builds_the_model_with_the_parsed_options_and_no_callbacks(monkeypatch, lora_file, full_argv):
    model = cli.Flux2Command.load(args_for(monkeypatch, full_argv))
    assert model.init_kwargs == expected_init_kwargs(lora_file)
    assert model.vae.decode_mode == "full_precision"
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_load_passes_the_small_decoder_and_a_custom_checkpoint(monkeypatch):
    args = args_for(monkeypatch, ["--prompt", "x", "--model", "/models/klein-base-9b-tune", "--small-decoder"])
    model = cli.Flux2Command.load(args)
    assert model.init_kwargs["small_decoder"] is True
    assert model.init_kwargs["model_path"] == "/models/klein-base-9b-tune"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["flux2-klein-base-9b"]


@pytest.mark.fast
def test_load_refuses_without_building(monkeypatch):
    args = args_for(monkeypatch, ["--prompt", "x", "--model", "flux2-klein-9b", "--small-text-encoder"])
    with pytest.raises(ValueError, match="only works with FLUX.2-klein-4B"):
        cli.Flux2Command.load(args)
    assert FakeFlux2Klein.instances == []


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt_and_resolved_dims(monkeypatch, ref_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.Flux2Command.load(args)
    image = cli.Flux2Command.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, ref_png), "prompt": "a different puffin"}]


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
def test_generate_applies_the_guidance_default_itself(monkeypatch, argv, expected):
    # A caller that skips main() gets 1.0, not the parser's None, on distilled and base Klein alike.
    args = args_for(monkeypatch, ["--prompt", "x", *argv])
    model = cli.Flux2Command.load(args)
    cli.Flux2Command.generate(model, args, 7, "x")
    assert model.generate_calls[0]["guidance"] == expected
    if "--guidance" not in argv:
        assert args.guidance is None  # the default is not written back into args


@pytest.mark.fast
def test_generate_always_runs_the_flow_match_scheduler(monkeypatch):
    # The shared parser accepts --scheduler; FLUX.2 runs the schedule it was trained with.
    args = args_for(monkeypatch, ["--prompt", "x", "--scheduler", "linear"])
    model = cli.Flux2Command.load(args)
    cli.Flux2Command.generate(model, args, 7, "x")
    assert model.generate_calls[0]["scheduler"] == "flow_match_euler_discrete"


@pytest.mark.fast
def test_the_steps_never_read_the_command_line(monkeypatch):
    # The negative-prompt refusal is keyed on sys.argv, so it belongs to main(); a caller of the
    # steps with a stale sys.argv sees neither the refusal nor a warning.
    args = args_for(monkeypatch, ["--prompt", "x", "--negative-prompt", "y"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cli.Flux2Command.validate(args)
        model = cli.Flux2Command.load(args)
        # A negative_prompt kwarg would fail here: the fake binds every call to the real signature.
        cli.Flux2Command.generate(model, args, 7, "x")
