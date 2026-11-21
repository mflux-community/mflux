import json
import sys

import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.moss_image.cli import moss_image_generate as cli
from mflux.models.moss_image.variants import MOSSImage
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


class FakeMOSSImage(FakeModel):
    real = MOSSImage


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeMOSSImage.instances.clear()
    monkeypatch.setattr(cli, "MOSSImage", FakeMOSSImage)
    reset_cli_globals(monkeypatch)
    yield
    FakeMOSSImage.instances.clear()


@pytest.fixture
def lora_file(tmp_path):
    # The parser resolves --lora before the model is built, so the file has to exist. The fake never reads it.
    path = tmp_path / "turbo.safetensors"
    path.touch()
    return path


@pytest.fixture
def full_argv(tmp_path, lora_file):
    # Every value differs from the parser default so a hardcoded default goes red. Sizes are
    # plain ints here and go to the model unchanged.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "320", "--height", "192",
        "--guidance", "4.5",
        "--negative-prompt", "blurry",
        "--step-cache-ratio", "0.25",
        "--uncond-skip-layers", "20", "24",
        "-q", "8",
        "--lora", str(lora_file), "0.7",
        "--bake-lora",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-moss-image", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv, model_class=FakeMOSSImage):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-moss-image", *argv])
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
        "guidance": 4.5,
        "negative_prompt": "blurry",
        "step_cache_ratio": 0.25,
        "uncond_skip_layers": (20, 24),
    }


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, full_argv, lora_file):
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["moss-image-sft"],
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.7],
        "bake_lora": True,
        "ssd_stream": False,
        "stream_lora": False,
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
    # 1024x1024, 50 steps, CFG 3, an empty negative prompt and no step reuse when no flag says
    # otherwise (the model card and the README).
    model = run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])
    call = model.generate_calls[0]
    assert (call["width"], call["height"], call["num_inference_steps"]) == (1024, 1024, 50)
    assert (call["guidance"], call["negative_prompt"], call["step_cache_ratio"]) == (3.0, "", None)
    assert call["uncond_skip_layers"] == (22,)
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]
    assert model.init_kwargs["quantize"] is None
    assert (model.init_kwargs["lora_paths"], model.init_kwargs["bake_lora"]) == (None, False)


@pytest.mark.fast
def test_main_warns_about_every_ignored_option_before_building_the_model(monkeypatch):
    warnings_at_build = []

    class OrderedFake(FakeMOSSImage):
        def __init__(self, **kwargs):
            warnings_at_build.append(len(warned))
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "MOSSImage", OrderedFake)
    with pytest.warns(UserWarning) as warned:
        run_main(monkeypatch, ["--prompt", "x", "--scheduler", "flow_match_euler_discrete"], OrderedFake)
    assert [str(w.message).split(";")[0] for w in warned] == ["--scheduler is ignored"]
    assert warnings_at_build == [1]


@pytest.mark.fast
@pytest.mark.parametrize(("base_argv", "key"), [([], "moss-image-sft"), (["--base-model", "moss-image-base"], "moss-image-base")])  # fmt: skip
def test_main_passes_a_custom_checkpoint_through(monkeypatch, base_argv, key):
    # A repo id is not a registry alias: parse_args moves it to model_path and the command's
    # own entry supplies the config, or the family entry --base-model names. Nothing is downloaded with the fake.
    model = run_main(monkeypatch, ["--prompt", "x", "--model", "someone/my-finetune", *base_argv])
    assert model.init_kwargs["model_path"] == "someone/my-finetune"
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS[key]


@pytest.mark.fast
def test_main_refuses_a_base_model_outside_the_family(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(sys, "argv", ["mflux-generate-moss-image", "--prompt", "x", "--model", "someone/my-finetune", "--base-model", "dev"])  # fmt: skip
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeMOSSImage.instances == []


@pytest.mark.fast
@pytest.mark.parametrize("name", ["moss-image-base", "moss-base", "OpenMOSS-Team/MOSS-Image-Base"])
def test_main_runs_the_base_model_with_the_sft_settings(monkeypatch, name):
    model = run_main(monkeypatch, ["--prompt", "x", "--model", name])
    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["moss-image-base"]
    call = model.generate_calls[0]
    assert (call["num_inference_steps"], call["guidance"], call["uncond_skip_layers"]) == (50, 3.0, (22,))


@pytest.mark.fast
def test_main_keeps_loras_as_runtime_layers_unless_asked_to_bake(monkeypatch, lora_file):
    model = run_main(monkeypatch, ["--prompt", "x", "--lora", str(lora_file), "0.6"])
    assert model.init_kwargs["lora_paths"] == [str(lora_file)]
    assert model.init_kwargs["lora_scales"] == [0.6]
    assert model.init_kwargs["bake_lora"] is False
    assert cli.build_parser().get_default("bake_lora") is cli.DEFAULT_BAKE_LORA is MOSSImage.DEFAULT_BAKE_LORA is False


@pytest.mark.fast
def test_main_lets_a_foreign_model_error_escape_without_building_anything(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(sys, "argv", ["mflux-generate-moss-image", "--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeMOSSImage.instances == []


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, capsys, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_hands_the_stepwise_handler_the_moss_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.moss_image.latent_creator import MOSSImageLatentCreator

    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is MOSSImageLatentCreator


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeMOSSImage, "generate_image", cancel_first)
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

    monkeypatch.setattr(FakeMOSSImage, "generate_image", boom)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-moss-image", *full_argv])
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

    monkeypatch.setattr(FakeMOSSImage, "generate_image", rewrite_after_first)
    run_main(
        monkeypatch, ["--prompt-file", str(prompt_file), "--seed", "7", "8", "--output", str(tmp_path / "out.png")]
    )
    assert seen == ["a", "b"]


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_returns_the_moss_config_without_building_or_reading_files(monkeypatch, tmp_path):
    args = args_for(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt")])
    assert cli.MOSSImageCommand.validate(args) is AVAILABLE_MODELS["moss-image-sft"]
    assert FakeMOSSImage.instances == []


@pytest.mark.fast
def test_validate_rejects_a_foreign_model_and_builds_nothing(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.MOSSImageCommand.validate(args)
    assert FakeMOSSImage.instances == []


@pytest.mark.fast
def test_load_rejects_a_foreign_model_without_building(monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--model", "dev"])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.MOSSImageCommand.load(args)
    assert FakeMOSSImage.instances == []


@pytest.mark.fast
def test_load_returns_a_bare_model(monkeypatch):
    model = cli.MOSSImageCommand.load(args_for(monkeypatch, ["--prompt", "x", "-q", "8"]))
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["moss-image-sft"],
        "quantize": 8,
        "model_path": None,
        "lora_paths": None,
        "lora_scales": None,
        "bake_lora": False,
        "ssd_stream": False,
        "stream_lora": False,
    }
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt(monkeypatch, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.MOSSImageCommand.load(args)
    image = cli.MOSSImageCommand.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7), "prompt": "a different puffin"}]


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--steps", "300"], "cannot run 300 steps"),
        (["--width", "1000"], "multiples of 16"),
    ],
)
def test_load_rejects_a_bad_step_count_or_size_before_building_the_model(monkeypatch, argv, message):
    args = args_for(monkeypatch, ["--prompt", "x", *argv])
    with pytest.raises(ValueError, match=message):
        cli.MOSSImageCommand.load(args)
    assert FakeMOSSImage.instances == []


@pytest.mark.fast
def test_main_passes_an_empty_skip_layer_flag_as_no_skipped_layers(monkeypatch):
    model = run_main(monkeypatch, ["--prompt", "x", "--uncond-skip-layers"])
    assert model.generate_calls[0]["uncond_skip_layers"] == ()


@pytest.mark.fast
def test_load_rejects_a_skip_layer_outside_the_transformer_before_building_the_model(monkeypatch):
    args = args_for(monkeypatch, ["--prompt", "x", "--uncond-skip-layers", "42"])
    with pytest.raises(ValueError, match=r"layer indices in \[0, 42\)"):
        cli.MOSSImageCommand.load(args)
    assert FakeMOSSImage.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(("recorded", "argv", "expected"), [([], [], ()), ([5, 9], [], (5, 9)), ([5], ["--uncond-skip-layers", "3"], (3,))])  # fmt: skip
def test_a_sidecar_replays_its_skip_layers_unless_the_flag_overrides_them(
    monkeypatch, tmp_path, recorded, argv, expected
):
    sidecar = tmp_path / "run.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a puffin", "seed": 7, "uncond_skip_layers": recorded}))
    model = run_main(monkeypatch, ["--config-from-metadata", str(sidecar), *argv])
    assert model.generate_calls[0]["uncond_skip_layers"] == expected


@pytest.mark.fast
@pytest.mark.parametrize(("recorded", "argv", "expected"), [({}, [], False), ({"bake_lora": True}, [], True), ({"bake_lora": True}, ["--no-bake-lora"], False), ({"bake_lora": False}, ["--bake-lora"], True)])  # fmt: skip
def test_a_sidecar_replays_its_bake_choice_unless_the_flag_overrides_it(
    monkeypatch, tmp_path, recorded, argv, expected
):
    sidecar = tmp_path / "run.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a puffin", "seed": 7, **recorded}))
    model = run_main(monkeypatch, ["--config-from-metadata", str(sidecar), *argv])
    assert model.init_kwargs["bake_lora"] is expected


@pytest.mark.fast
def test_a_sidecar_with_a_non_boolean_bake_choice_fails_before_the_model_loads(monkeypatch, tmp_path):
    sidecar = tmp_path / "run.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a puffin", "bake_lora": "false"}))
    monkeypatch.setattr(sys, "argv", ["mflux-generate-moss-image", "--config-from-metadata", str(sidecar)])
    with pytest.raises(SystemExit):
        cli.main()
    assert FakeMOSSImage.instances == []


@pytest.mark.fast
def test_a_sidecar_with_bad_skip_layers_fails_before_the_model_loads(monkeypatch, tmp_path):
    sidecar = tmp_path / "run.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a puffin", "uncond_skip_layers": "22"}))
    monkeypatch.setattr(sys, "argv", ["mflux-generate-moss-image", "--config-from-metadata", str(sidecar)])
    with pytest.raises(SystemExit):
        cli.main()
    assert FakeMOSSImage.instances == []
