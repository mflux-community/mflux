import sys

import mlx.core as mx
import PIL.Image
import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.z_image.cli import z_image_generate as cli
from mflux.models.z_image.variants.z_image import ZImage
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


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
    # Every value differs from the parser default so a hardcoded default goes red. Guidance is
    # above 1.0 so the base model encodes the negative prompt and main() warns about nothing.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "320", "--height", "192",
        "--image", str(ref_png), "0.55",
        "--guidance", "2.5",
        "--negative-prompt", "blurry",
        "--scheduler", "linear",
        "--pid-decode", "--pid-degrade-sigma", "0.2",
        "--lora", str(lora_file), "0.5",
        "-q", "8",
        "--compute-precision", "float16",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image", *argv])
    cli.main()
    assert len(FakeZImage.instances) == 1
    return FakeZImage.instances[0]


def expected_generate_call(seed, ref_png):
    # This parser takes plain ints (no supports_dimension_scale_factor), so the sizes pass through.
    return {
        "seed": seed,
        "prompt": "a puffin",
        "width": 320,
        "height": 192,
        "guidance": 2.5,
        "image_path": ref_png,
        "num_inference_steps": 3,
        "image_strength": 0.55,
        "scheduler": "linear",
        "negative_prompt": "blurry",
        "pid_decode": True,
        "pid_degrade_sigma": 0.2,
    }


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, ref_png, lora_file, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["z-image"],
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": True,
        "compute_precision": mx.float16,
    }
    assert model.generate_calls == [expected_generate_call(7, ref_png), expected_generate_call(8, ref_png)]
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_defaults_the_scheduler_only_when_the_flag_is_absent(monkeypatch):
    # The command's own default lives in main() because it reads sys.argv; an explicit
    # --scheduler, even the parser default, wins.
    assert run_main(monkeypatch, ["--prompt", "x"]).generate_calls[0]["scheduler"] == "flow_match_euler_discrete"
    FakeZImage.instances.clear()
    assert run_main(monkeypatch, ["--prompt", "x", "--scheduler", "linear"]).generate_calls[0]["scheduler"] == "linear"


@pytest.mark.fast
def test_main_runs_a_family_sibling_and_warns_before_building_it(monkeypatch):
    warnings_at_build = []

    class OrderedFake(FakeZImage):
        def __init__(self, **kwargs):
            warnings_at_build.append(len(warned))
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "ZImage", OrderedFake)
    with pytest.warns(UserWarning) as warned:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "mflux-generate-z-image",
                "--prompt",
                "x",
                "--model",
                "z-image-turbo",
                "--guidance",
                "3",
                "--negative-prompt",
                "y",
            ],
        )
        cli.main()
    assert sorted(str(w.message).split(";")[0] for w in warned) == [
        "--guidance is ignored",
        "--negative-prompt is ignored",
    ]
    assert warnings_at_build == [2]  # both warnings were out before the model was built
    assert OrderedFake.instances[0].init_kwargs["model_config"] is AVAILABLE_MODELS["z-image-turbo"]


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image", "--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeZImage.instances == []


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeZImage, "generate_image", cancel_first)
    run_main(monkeypatch, full_argv)
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
    assert "Peak MLX memory: " in out
    assert len(FakeZImage.instances[0].generate_calls) == 1


@pytest.mark.fast
def test_main_prints_a_missing_prompt_file_and_generates_nothing(monkeypatch, capsys, tmp_path):
    model = run_main(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt")])
    assert "Prompt file does not exist" in capsys.readouterr().out
    assert model.generate_calls == []


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_accepts_the_family_sibling_and_builds_nothing(monkeypatch, tmp_path):
    args = args_for(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt"), "--model", "z-image-turbo"])
    assert cli.ZImageCommand.validate(args) == AVAILABLE_MODELS["z-image-turbo"]
    assert FakeZImage.instances == []


@pytest.mark.fast
def test_validate_rejects_a_foreign_model(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.ZImageCommand.validate(args_for(monkeypatch, ["--prompt", "x", "--model", "dev"]))


@pytest.mark.fast
def test_load_returns_a_bare_model(monkeypatch, lora_file):
    model = cli.ZImageCommand.load(args_for(monkeypatch, ["--prompt", "x", "--lora", str(lora_file), "0.5"]))
    assert model.init_kwargs["lora_paths"] == [str(lora_file)]
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt(monkeypatch, ref_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.ZImageCommand.load(args)
    image = cli.ZImageCommand.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, ref_png), "prompt": "a different puffin"}]


@pytest.mark.fast
def test_load_passes_a_custom_checkpoint_through_as_its_family_member(monkeypatch):
    # A repo id is not a registry alias: parse_args moves it to model_path and the family
    # entry it names supplies the config. Nothing is downloaded with the fake.
    model = cli.ZImageCommand.load(args_for(monkeypatch, ["--prompt", "x", "--model", "someone/Z-Image-Turbo-4bit"]))
    assert model.init_kwargs["model_path"] == "someone/Z-Image-Turbo-4bit"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["z-image-turbo"]


@pytest.mark.fast
def test_validate_rejects_a_base_model_outside_the_family_before_building(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "someone/my-finetune", "--base-model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.ZImageCommand.load(args)
    assert FakeZImage.instances == []


@pytest.mark.fast
def test_main_wires_the_stepwise_handler_with_the_z_image_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.z_image.latent_creator import ZImageLatentCreator

    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is ZImageLatentCreator
