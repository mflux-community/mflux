import sys
from pathlib import Path

import PIL.Image
import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.z_image.cli import z_image_turbo_generate_controlnet as cli
from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
from mflux.models.z_image.variants.controlnet.z_image_turbo_controlnet import ZImageTurboControlnet
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

pytestmark = pytest.mark.filterwarnings("ignore:--(guidance|negative-prompt) is ignored:UserWarning")


class FakeControlnet(FakeModel):
    real = ZImageTurboControlnet


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeControlnet.instances.clear()
    monkeypatch.setattr(cli, "ZImageTurboControlnet", FakeControlnet)
    reset_cli_globals(monkeypatch)
    yield
    FakeControlnet.instances.clear()


@pytest.fixture
def control_png(tmp_path):
    path = tmp_path / "edges.png"
    PIL.Image.new("RGB", (64, 32)).save(path)
    return path


@pytest.fixture
def lora_file(tmp_path):
    path = tmp_path / "style.safetensors"
    path.write_bytes(b"")
    return path


@pytest.fixture
def full_argv(tmp_path, control_png, lora_file):
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "320", "--height", "192",
        "--control", f"canny:{control_png}:0.6",
        "--controlnet-strength", "0.8",
        "--scheduler", "flow_match_euler_discrete",
        "--lora", str(lora_file), "0.5",
        "-q", "8",
        "--float32",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-controlnet", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-controlnet", *argv])
    cli.main()
    assert len(FakeControlnet.instances) == 1
    return FakeControlnet.instances[0]


def expected_generate_call(seed, control_png):
    return {
        "seed": seed,
        "prompt": "a puffin",
        "width": 320,
        "height": 192,
        "scheduler": "flow_match_euler_discrete",
        "num_inference_steps": 3,
        "controlnet_strength": 0.8,
        "controls": [ControlSpec(type=ControlType("canny"), image_path=Path(control_png), strength=0.6)],
    }


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, control_png, lora_file, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["z-image-turbo-controlnet-union-2.1"],
        "quantize": 8,
        "model_path": None,
        "float32": True,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
    }
    assert model.generate_calls == [expected_generate_call(7, control_png), expected_generate_call(8, control_png)]
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_rejects_a_model_without_a_controlnet_before_building_it(monkeypatch, capsys, control_png):
    monkeypatch.setattr(sys, "argv", ["x", "--prompt", "p", "--control", f"canny:{control_png}", "--model", "dev"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert "is not a Union ControlNet model" in capsys.readouterr().err
    assert FakeControlnet.instances == []


@pytest.mark.fast
def test_main_rejects_a_malformed_control_spec_before_building_it(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--prompt", "p", "--control", "canny"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert "Expected format type:path[:strength]" in capsys.readouterr().err
    assert FakeControlnet.instances == []


@pytest.mark.fast
def test_main_keeps_the_traceback_for_a_model_the_registry_cannot_place(monkeypatch, control_png):
    from mflux.utils.exceptions import ModelConfigError

    # The ControlNet check is exit 2, but a --model the registry cannot place kept its
    # traceback before the split and still does.
    monkeypatch.setattr(
        sys, "argv", ["x", "--prompt", "p", "--control", f"canny:{control_png}", "--model", "someone/my-controlnet"]
    )
    with pytest.raises(ModelConfigError, match="Cannot infer base_model from someone/my-controlnet"):
        cli.main()
    assert FakeControlnet.instances == []


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeControlnet, "generate_image", cancel_first)
    run_main(monkeypatch, full_argv)
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
    assert "Peak MLX memory: " in out
    assert len(FakeControlnet.instances[0].generate_calls) == 1


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_returns_the_controlnet_config_without_building_or_reading_files(monkeypatch, tmp_path):
    args = args_for(monkeypatch, ["--prompt", "p", "--control", f"canny:{tmp_path / 'missing.png'}"])
    assert cli.ZImageTurboControlnetCommand.validate(args) == AVAILABLE_MODELS["z-image-turbo-controlnet-union-2.1"]
    assert FakeControlnet.instances == []


@pytest.mark.fast
def test_validate_raises_for_a_model_without_a_controlnet_and_for_a_bad_spec(monkeypatch, control_png):
    with pytest.raises(ValueError, match="is not a Union ControlNet model"):
        cli.ZImageTurboControlnetCommand.validate(
            args_for(monkeypatch, ["--prompt", "p", "--control", f"canny:{control_png}", "--model", "dev"])
        )
    with pytest.raises(ValueError, match="Missing image path"):
        cli.ZImageTurboControlnetCommand.validate(args_for(monkeypatch, ["--prompt", "p", "--control", "canny::0.5"]))


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_parsed_controls(monkeypatch, control_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.ZImageTurboControlnetCommand.load(args)
    image = cli.ZImageTurboControlnetCommand.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, control_png), "prompt": "a different puffin"}]


@pytest.mark.fast
def test_load_returns_a_bare_model(monkeypatch, control_png, lora_file):
    args = args_for(
        monkeypatch, ["--prompt", "p", "--control", f"canny:{control_png}", "--lora", str(lora_file), "0.5"]
    )
    model = cli.ZImageTurboControlnetCommand.load(args)
    assert model.init_kwargs["lora_paths"] == [str(lora_file)]
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]


@pytest.mark.fast
def test_main_wires_the_stepwise_handler_with_the_z_image_latent_creator(monkeypatch, tmp_path, control_png):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.z_image.latent_creator import ZImageLatentCreator

    model = run_main(
        monkeypatch,
        ["--prompt", "x", "--control", f"canny:{control_png}", "--stepwise-image-output-dir", str(tmp_path / "steps")],
    )
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is ZImageLatentCreator


@pytest.mark.fast
def test_load_rejects_a_model_without_a_controlnet_before_building_it(monkeypatch, control_png):
    with pytest.raises(ValueError, match="is not a Union ControlNet model"):
        cli.ZImageTurboControlnetCommand.load(
            args_for(monkeypatch, ["--prompt", "p", "--control", f"canny:{control_png}", "--model", "dev"])
        )
    assert FakeControlnet.instances == []
