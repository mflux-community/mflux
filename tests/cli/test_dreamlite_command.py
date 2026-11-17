import json
import sys

import mlx.core as mx
import pytest
from PIL import Image

from mflux.cli.capabilities import describe_command
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.dreamlite.cli import (
    dreamlite_edit_generate as edit_cli,
    dreamlite_generate as cli,
)
from mflux.models.dreamlite.latent_creator.dreamlite_latent_creator import DreamLiteLatentCreator
from mflux.models.dreamlite.variants import DreamLite, DreamLiteEdit
from mflux.utils.exceptions import ModelConfigError
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


class FakeDreamLite(FakeModel):
    real = DreamLite


class FakeDreamLiteEdit(FakeModel):
    real = DreamLiteEdit


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeDreamLite.instances.clear()
    FakeDreamLiteEdit.instances.clear()
    monkeypatch.setattr(cli, "DreamLite", FakeDreamLite)
    monkeypatch.setattr(edit_cli, "DreamLiteEdit", FakeDreamLiteEdit)
    reset_cli_globals(monkeypatch)
    yield
    FakeDreamLite.instances.clear()
    FakeDreamLiteEdit.instances.clear()


@pytest.fixture
def source_image(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (640, 480)).save(path)
    return path


class CommandRunner:
    @staticmethod
    def args_for(monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite", *argv])
        return cli.build_parser().parse_args()

    @staticmethod
    def run_main(monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite", *argv])
        cli.main()
        assert len(FakeDreamLite.instances) == 1
        return FakeDreamLite.instances[0]

    @staticmethod
    def run_edit(monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite-edit", *argv])
        edit_cli.main()
        assert len(FakeDreamLiteEdit.instances) == 1
        return FakeDreamLiteEdit.instances[0]


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path):
    model = CommandRunner.run_main(
        monkeypatch,
        ["--prompt", "a puffin", "--seed", "7", "8", "--steps", "8", "--width", "320", "--height", "192",
         "--guidance", "2", "--negative-prompt", "blur", "-q", "8", "--metadata",
         "--output", str(tmp_path / "out.png")],
    )  # fmt: skip

    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["dreamlite"],
        "quantize": 8,
        "model_path": None,
        "compute_precision": None,
    }
    expected = {"prompt": "a puffin", "negative_prompt": "blur", "width": 320, "height": 192, "num_inference_steps": 8, "guidance": 2.0}  # fmt: skip
    assert model.generate_calls == [{"seed": 7, **expected}, {"seed": 8, **expected}]
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_runs_the_defaults_of_the_base_model(monkeypatch, tmp_path):
    # 1024x1024 and 28 steps; guidance stays None so the model applies its own 3.5
    model = CommandRunner.run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])

    call = model.generate_calls[0]
    assert (call["width"], call["height"], call["num_inference_steps"], call["guidance"]) == (1024, 1024, 28, None)
    assert DreamLite.DEFAULT_GUIDANCE == 3.5


@pytest.mark.fast
def test_mobile_runs_four_steps(monkeypatch):
    model = CommandRunner.run_main(monkeypatch, ["--prompt", "x", "--model", "dreamlite-mobile"])

    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["dreamlite-mobile"]
    assert model.generate_calls[0]["num_inference_steps"] == 4
    assert AVAILABLE_MODELS["dreamlite-mobile"].supports_guidance is False


@pytest.mark.fast
@pytest.mark.parametrize("key", ["dreamlite", "dreamlite-mobile"])
def test_every_alias_runs_its_model(monkeypatch, key):
    for name in AVAILABLE_MODELS[key].aliases:
        FakeDreamLite.instances.clear()
        assert CommandRunner.run_main(monkeypatch, ["--prompt", "x", "--model", name]).init_kwargs["model_config"] is AVAILABLE_MODELS[key]  # fmt: skip


@pytest.mark.fast
def test_main_passes_a_custom_checkpoint_through(monkeypatch):
    model = CommandRunner.run_main(
        monkeypatch, ["--prompt", "x", "--model", "/models/my-q4", "--base-model", "dreamlite-mobile"]
    )

    assert model.init_kwargs["model_path"] == "/models/my-q4"
    assert model.init_kwargs["model_config"].model_name == "carlofkl/DreamLite-mobile"


@pytest.mark.fast
def test_main_refuses_a_foreign_model_without_building_anything(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite", "--prompt", "x", "--model", "dev"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "only accepts the aliases" in capsys.readouterr().err
    with pytest.raises(ModelConfigError):
        cli.DreamLiteCommand.load(CommandRunner.args_for(monkeypatch, ["--prompt", "x", "--model", "dev"]))
    assert FakeDreamLite.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--width", "1000"], "multiples of 32"),
        (["--height", "1040"], "multiples of 32"),
        (["--width", "0"], "must be positive"),
    ],
)
def test_main_rejects_a_bad_size_before_building_the_model(monkeypatch, capsys, argv, message):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite", "--prompt", "x", *argv])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert FakeDreamLite.instances == []


@pytest.mark.fast
def test_main_warns_about_every_ignored_option_before_building_the_model(monkeypatch):
    warnings_at_build = []

    class OrderedFake(FakeDreamLite):
        def __init__(self, **kwargs):
            warnings_at_build.append(len(warned))
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "DreamLite", OrderedFake)
    monkeypatch.setattr(
        sys, "argv", ["mflux-generate-dreamlite", "--prompt", "x", "--scheduler", "linear", "--vae-tiling"]
    )
    with pytest.warns(UserWarning) as warned:
        cli.main()

    assert sorted(str(w.message).split(";")[0] for w in warned) == ["--scheduler is ignored", "--vae-tiling is ignored"]
    assert warnings_at_build == [2]


@pytest.mark.fast
def test_mobile_warns_about_guidance_and_about_a_negative_prompt_it_will_not_use(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite", "--prompt", "x", "--model", "dreamlite-mobile", "--guidance", "3"])  # fmt: skip
    with pytest.warns(UserWarning, match="distilled to run without guidance; the value is ignored"):
        cli.main()

    FakeDreamLite.instances.clear()
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite", "--prompt", "x", "--model", "dreamlite-mobile", "--negative-prompt", "blur"])  # fmt: skip
    with pytest.warns(UserWarning, match="--negative-prompt is ignored at guidance 1.0"):
        cli.main()


@pytest.mark.fast
def test_the_base_model_takes_a_negative_prompt_without_a_warning(monkeypatch, recwarn):
    CommandRunner.run_main(monkeypatch, ["--prompt", "x", "--negative-prompt", "blur"])

    assert [str(w.message) for w in recwarn.list] == []


@pytest.mark.fast
def test_main_hands_the_stepwise_handler_the_dreamlite_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler

    model = CommandRunner.run_main(
        monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")]
    )

    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is DreamLiteLatentCreator


@pytest.mark.fast
def test_load_and_generate_are_the_steps_a_python_caller_uses(monkeypatch):
    args = CommandRunner.args_for(monkeypatch, ["--prompt", "x", "-q", "4", "--steps", "3"])

    model = cli.DreamLiteCommand.load(args)
    image = cli.DreamLiteCommand.generate(model, args, 5, "a puffin")

    assert model.init_kwargs["quantize"] == 4
    assert image.saves == []
    assert model.generate_calls == [
        {
            "seed": 5,
            "prompt": "a puffin",
            "negative_prompt": "",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 3,
            "guidance": None,
        }  # fmt: skip
    ]


@pytest.mark.fast
def test_edit_call_sequence_is_pinned(monkeypatch, tmp_path, source_image):
    model = CommandRunner.run_edit(
        monkeypatch,
        ["--prompt", "make it snowy", "--image-paths", str(source_image), "--seed", "7", "--image-guidance", "1.5",
         "--negative-prompt", "blur", "-q", "8", "--output", str(tmp_path / "out.png")],
    )  # fmt: skip

    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["dreamlite"],
        "quantize": 8,
        "model_path": None,
        "compute_precision": None,
    }
    assert model.generate_calls == [
        {
            "seed": 7,
            "prompt": "make it snowy",
            "image_path": source_image,
            "negative_prompt": "blur",
            "width": 640,
            "height": 480,
            "guidance": None,
            "image_guidance": 1.5,
            "num_inference_steps": 28,
        }
    ]
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]


@pytest.mark.fast
def test_an_edit_can_be_scaled_from_the_source_size_or_given_a_size(monkeypatch, source_image):
    scaled = CommandRunner.run_edit(
        monkeypatch, ["--prompt", "x", "--image-paths", str(source_image), "--width", "2x", "--height", "2x"]
    )
    assert (scaled.generate_calls[0]["width"], scaled.generate_calls[0]["height"]) == (1280, 960)

    FakeDreamLiteEdit.instances.clear()
    sized = CommandRunner.run_edit(
        monkeypatch, ["--prompt", "x", "--image-paths", str(source_image), "--width", "512", "--height", "512"]
    )
    assert (sized.generate_calls[0]["width"], sized.generate_calls[0]["height"]) == (512, 512)


@pytest.mark.fast
def test_an_edit_on_mobile_runs_four_steps(monkeypatch, source_image):
    model = CommandRunner.run_edit(
        monkeypatch, ["--prompt", "x", "--image-paths", str(source_image), "--model", "dreamlite-mobile"]
    )

    assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["dreamlite-mobile"]
    assert model.generate_calls[0]["num_inference_steps"] == 4


@pytest.mark.fast
def test_an_edit_takes_exactly_one_image(monkeypatch, capsys, source_image):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite-edit", "--prompt", "x", "--image-paths", str(source_image), str(source_image)])  # fmt: skip

    with pytest.raises(SystemExit) as exit_info:
        edit_cli.main()

    assert exit_info.value.code == 2
    assert "takes one image" in capsys.readouterr().err
    assert FakeDreamLiteEdit.instances == []


@pytest.mark.fast
@pytest.mark.parametrize("argv", [["--width", "-512"], ["--height", "0"]])
def test_an_edit_rejects_a_size_that_is_not_positive_before_building_the_model(monkeypatch, capsys, source_image, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite-edit", "--prompt", "x", "--image-paths", str(source_image), *argv])  # fmt: skip

    with pytest.raises(SystemExit) as exit_info:
        edit_cli.main()

    assert exit_info.value.code == 2
    assert "must be positive" in capsys.readouterr().err
    assert FakeDreamLiteEdit.instances == []


@pytest.mark.fast
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_an_image_guidance_that_is_not_finite_is_a_usage_error(monkeypatch, capsys, source_image, value):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite-edit", "--prompt", "x", "--image-paths", str(source_image), "--image-guidance", value])  # fmt: skip

    with pytest.raises(SystemExit) as exit_info:
        edit_cli.main()

    assert exit_info.value.code == 2
    assert "must be a finite number > 0" in capsys.readouterr().err
    assert FakeDreamLiteEdit.instances == []


@pytest.mark.fast
def test_image_guidance_on_mobile_is_reported_as_unused(monkeypatch, source_image):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite-edit", "--prompt", "x", "--image-paths", str(source_image), "--model", "dreamlite-mobile", "--image-guidance", "1.5"])  # fmt: skip

    with pytest.warns(UserWarning, match="--image-guidance is ignored at guidance 1.0"):
        edit_cli.main()


@pytest.mark.fast
def test_a_sidecar_replays_the_image_guidance_unless_the_flag_is_given(monkeypatch, tmp_path, source_image):
    sidecar = tmp_path / "run.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "make it snowy", "seed": 7, "image_paths": [str(source_image)], "image_guidance": 1.5}))  # fmt: skip

    replayed = CommandRunner.run_edit(monkeypatch, ["--config-from-metadata", str(sidecar)])
    assert replayed.generate_calls[0]["image_guidance"] == 1.5
    assert replayed.generate_calls[0]["image_path"] == source_image

    FakeDreamLiteEdit.instances.clear()
    overridden = CommandRunner.run_edit(monkeypatch, ["--config-from-metadata", str(sidecar), "--image-guidance", "2"])
    assert overridden.generate_calls[0]["image_guidance"] == 2.0


@pytest.mark.fast
@pytest.mark.parametrize("value", [-1, "nan", "inf", float("nan"), float("inf")])
def test_a_sidecar_with_a_bad_image_guidance_is_a_usage_error(monkeypatch, tmp_path, source_image, capsys, value):
    sidecar = tmp_path / "run.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "x", "seed": 7, "image_paths": [str(source_image)], "image_guidance": value}))  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite-edit", "--config-from-metadata", str(sidecar)])

    with pytest.raises(SystemExit) as exit_info:
        edit_cli.main()

    assert exit_info.value.code == 2
    assert "image_guidance in --config-from-conf" in capsys.readouterr().err
    assert FakeDreamLiteEdit.instances == []


@pytest.mark.fast
def test_compute_precision_reaches_the_model_in_both_commands(monkeypatch, source_image):
    generate = CommandRunner.run_main(monkeypatch, ["--prompt", "x", "--compute-precision", "float16"])
    edit = CommandRunner.run_edit(monkeypatch, ["--prompt", "x", "--image-paths", str(source_image), "--compute-precision", "float16"])  # fmt: skip

    assert generate.init_kwargs["compute_precision"] == mx.float16
    assert edit.init_kwargs["compute_precision"] == mx.float16
    assert "UNet" in cli.COMPUTE_PRECISION_HELP


@pytest.mark.fast
def test_a_sidecar_replays_the_compute_precision(monkeypatch, tmp_path):
    sidecar = tmp_path / "run.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a puffin", "seed": 7, "compute_precision": "float16"}))

    model = CommandRunner.run_main(monkeypatch, ["--config-from-metadata", str(sidecar)])

    assert model.init_kwargs["compute_precision"] == mx.float16


@pytest.mark.fast
def test_both_commands_bound_the_mlx_cache_unless_told_otherwise(monkeypatch, source_image):
    # Unbounded, the cache doubles the memory of the process and buys no speed
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite", "--prompt", "x"])
    assert cli.build_parser().parse_args().mlx_cache_limit_gb == 1.0
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite-edit", "--prompt", "x", "--image-paths", str(source_image)])  # fmt: skip
    assert edit_cli.build_parser().parse_args().mlx_cache_limit_gb == 1.0
    monkeypatch.setattr(sys, "argv", ["mflux-generate-dreamlite", "--prompt", "x", "--mlx-cache-limit-gb", "8"])
    assert cli.build_parser().parse_args().mlx_cache_limit_gb == 8.0


@pytest.mark.fast
def test_main_applies_the_cache_limit(monkeypatch):
    limits = []
    monkeypatch.setattr(mx, "set_cache_limit", lambda value: limits.append(value))

    CommandRunner.run_main(monkeypatch, ["--prompt", "x"])

    assert limits == [1000**3]


@pytest.mark.fast
def test_capabilities_name_the_ignored_options_and_the_edit_flags():
    generate = {option["flag"]: option for option in describe_command("mflux-generate-dreamlite", cli.__name__)["options"]}  # fmt: skip
    edit = {option["flag"]: option for option in describe_command("mflux-generate-dreamlite-edit", edit_cli.__name__)["options"]}  # fmt: skip

    assert generate["--scheduler"]["status"] == "ignored"
    assert generate["--vae-tiling"]["status"] == "ignored"
    assert generate["--negative-prompt"]["status"] != "ignored"
    for flag in ("--image-paths", "--image-guidance", "--lora"):
        assert flag not in generate, flag
    assert "--image-paths" in edit and "--image-guidance" in edit
    assert "--lora" not in edit
