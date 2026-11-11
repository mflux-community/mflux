import json
import sys

import pytest

from mflux.cli.capabilities import describe_command
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.sensenova.cli import sensenova_generate as cli
from mflux.models.sensenova.variants import SenseNova
from mflux.utils.exceptions import ModelConfigError
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


class FakeSenseNova(FakeModel):
    real = SenseNova


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeSenseNova.instances.clear()
    monkeypatch.setattr(cli, "SenseNova", FakeSenseNova)
    reset_cli_globals(monkeypatch)
    yield
    FakeSenseNova.instances.clear()


@pytest.fixture
def lora_file(tmp_path):
    path = tmp_path / "lora.safetensors"
    path.touch()
    return path


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-sensenova", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-sensenova", *argv])
    cli.main()
    assert len(FakeSenseNova.instances) == 1
    return FakeSenseNova.instances[0]


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, lora_file):
    model = run_main(
        monkeypatch,
        ["--prompt", "a puffin", "--seed", "7", "8", "--steps", "8", "--width", "320", "--height", "192",
         "--guidance", "1", "-q", "8", "--make-conf", "--lora", str(lora_file), "0.7", "--bake-lora",
         "--output", str(tmp_path / "out.png")],
    )  # fmt: skip

    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["sensenova-u1.5"],
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.7],
        "bake_lora": True,
    }
    expected = {"prompt": "a puffin", "width": 320, "height": 192, "num_inference_steps": 8, "guidance": 1.0}
    assert model.generate_calls == [{"seed": 7, **expected}, {"seed": 8, **expected}]
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_runs_the_defaults_of_the_base_model(monkeypatch, tmp_path):
    # 1024x1024, 50 steps and CFG 4 when no flag says otherwise (the model card and the README)
    model = run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])

    call = model.generate_calls[0]
    assert (call["width"], call["height"], call["num_inference_steps"], call["guidance"]) == (1024, 1024, 50, 4.0)
    assert "negative_prompt" not in call
    assert (model.init_kwargs["quantize"], model.init_kwargs["lora_paths"]) == (None, None)


@pytest.mark.fast
@pytest.mark.parametrize("name", AVAILABLE_MODELS["sensenova-u1.5"].aliases)
def test_every_alias_runs_the_model(monkeypatch, name):
    assert run_main(monkeypatch, ["--prompt", "x", "--model", name]).init_kwargs["model_config"] is AVAILABLE_MODELS["sensenova-u1.5"]  # fmt: skip


@pytest.mark.fast
def test_main_keeps_loras_as_runtime_layers_unless_asked_to_bake(monkeypatch, lora_file):
    model = run_main(monkeypatch, ["--prompt", "x", "--lora", str(lora_file)])

    assert model.init_kwargs["bake_lora"] is False
    assert cli.build_parser().get_default("bake_lora") is cli.DEFAULT_BAKE_LORA is SenseNova.DEFAULT_BAKE_LORA is False


@pytest.mark.fast
def test_the_lora_does_not_change_the_step_count_or_guidance(monkeypatch, lora_file):
    # The official 8-step LoRA carries no step metadata: --steps 8 --guidance 1 are the user's to give
    model = run_main(monkeypatch, ["--prompt", "x", "--lora", str(lora_file)])

    assert (model.generate_calls[0]["num_inference_steps"], model.generate_calls[0]["guidance"]) == (50, 4.0)


@pytest.mark.fast
def test_main_passes_a_custom_checkpoint_through(monkeypatch):
    model = run_main(monkeypatch, ["--prompt", "x", "--model", "/models/sensenova-q4"])

    assert model.init_kwargs["model_path"] == "/models/sensenova-q4"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["sensenova-u1.5"]


@pytest.mark.fast
def test_main_refuses_a_foreign_model_without_building_anything(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-sensenova", "--prompt", "x", "--model", "dev"])

    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeSenseNova.instances == []


@pytest.mark.fast
def test_main_rejects_a_negative_prompt_before_building_the_model(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-sensenova", "--prompt", "x", "--negative-prompt", "blur"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "--negative-prompt is not supported" in capsys.readouterr().err
    assert FakeSenseNova.instances == []


@pytest.mark.fast
def test_a_sidecar_negative_prompt_is_not_the_users_and_is_not_passed_on(monkeypatch, tmp_path):
    sidecar = tmp_path / "run.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a puffin", "seed": 7, "negative_prompt": "blur"}))

    model = run_main(monkeypatch, ["--config-from-metadata", str(sidecar)])

    assert "negative_prompt" not in model.generate_calls[0]


@pytest.mark.fast
@pytest.mark.parametrize("argv", [["--width", "1000"], ["--height", "1040"]])
def test_main_rejects_a_size_that_is_not_a_multiple_of_32_before_building_the_model(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-sensenova", "--prompt", "x", *argv])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "multiples of 32" in capsys.readouterr().err
    assert FakeSenseNova.instances == []


@pytest.mark.fast
def test_main_accepts_a_size_outside_the_trained_buckets_without_a_warning(monkeypatch, recwarn):
    run_main(monkeypatch, ["--prompt", "x", "--width", "1024", "--height", "1024"])

    assert [str(w.message) for w in recwarn.list] == []


@pytest.mark.fast
def test_main_warns_about_every_ignored_option_before_building_the_model(monkeypatch):
    warnings_at_build = []

    class OrderedFake(FakeSenseNova):
        def __init__(self, **kwargs):
            warnings_at_build.append(len(warned))
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "SenseNova", OrderedFake)
    monkeypatch.setattr(
        sys, "argv", ["mflux-generate-sensenova", "--prompt", "x", "--scheduler", "linear", "--vae-tiling"]
    )
    with pytest.warns(UserWarning) as warned:
        cli.main()

    assert sorted(str(w.message).split(";")[0] for w in warned) == ["--scheduler is ignored", "--vae-tiling is ignored"]
    assert warnings_at_build == [2]


@pytest.mark.fast
def test_main_hands_the_stepwise_handler_the_sensenova_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.sensenova.latent_creator.sensenova_latent_creator import SenseNovaLatentCreator

    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])

    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is SenseNovaLatentCreator


@pytest.mark.fast
def test_main_reports_memory_after_the_run(monkeypatch, capsys):
    run_main(monkeypatch, ["--prompt", "x"])

    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_load_and_generate_are_the_steps_a_python_caller_uses(monkeypatch):
    args = args_for(monkeypatch, ["--prompt", "x", "-q", "4", "--steps", "3"])

    model = cli.SenseNovaCommand.load(args)
    image = cli.SenseNovaCommand.generate(model, args, 5, "a puffin")

    assert model.init_kwargs["quantize"] == 4
    assert image.saves == []
    assert model.generate_calls == [
        {"seed": 5, "prompt": "a puffin", "width": 1024, "height": 1024, "num_inference_steps": 3, "guidance": 4.0}
    ]


@pytest.mark.fast
def test_validate_rejects_a_bad_size_and_a_foreign_model_before_building_anything(monkeypatch):
    with pytest.raises(ValueError, match="multiples of 32"):
        cli.SenseNovaCommand.validate(args_for(monkeypatch, ["--prompt", "x", "--width", "1000"]))
    with pytest.raises(ModelConfigError):
        cli.SenseNovaCommand.load(args_for(monkeypatch, ["--prompt", "x", "--model", "dev"]))
    assert FakeSenseNova.instances == []


@pytest.mark.fast
def test_capabilities_name_the_rejected_and_ignored_options():
    command = describe_command("mflux-generate-sensenova", cli.__name__)
    options = {option["flag"]: option for option in command["options"]}

    assert options["--negative-prompt"]["status"] == "rejected"
    assert options["--scheduler"]["status"] == "ignored"
    assert options["--lora"]["status"] == "honored"
    assert options["--bake-lora"]["parser_default"] is False
    assert options["--guidance"]["parser_default"] == 4.0
    for flag in ("--image-path", "--image", "--image-paths", "--controlnet-image-path"):
        assert flag not in options, flag
