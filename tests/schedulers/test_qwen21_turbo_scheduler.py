import sys

import mlx.core as mx
import pytest

from mflux.cli.defaults.defaults import model_inference_steps
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers import SCHEDULER_REGISTRY
from mflux.models.qwen21.cli import qwen21_controlnet_generate, qwen21_edit_generate, qwen21_generate
from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler

# sample_sigmas from Qwen/Qwen-Image-2.1-Turbo/model_index.json
MODEL_INDEX_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def _turbo_config(width=1024, height=1024, steps=8, scheduler="linear"):
    model_config = ModelConfig.qwen_image_21_turbo()
    return Config(
        model_config=model_config,
        num_inference_steps=steps,
        width=width,
        height=height,
        scheduler=Qwen21TurboScheduler.for_model(model_config, scheduler),
    )


@pytest.mark.fast
def test_registry_entry():
    config = ModelConfig.from_name("qwen-2.1-turbo")
    assert config is ModelConfig.qwen_image_21_turbo()
    assert config.model_name == "Qwen/Qwen-Image-2.1-Turbo"
    assert config.supports_guidance is False
    assert SCHEDULER_REGISTRY["qwen21_turbo"] is Qwen21TurboScheduler


@pytest.mark.fast
@pytest.mark.parametrize(
    "name", ["qwen-image-2.1-turbo", "qwen-2.1-turbo", "qwen-image-turbo-21", "Qwen/Qwen-Image-2.1-Turbo"]
)
def test_default_step_count_is_eight(name):
    assert model_inference_steps(name) == 8


@pytest.mark.fast
@pytest.mark.parametrize("width,height", [(1024, 1024), (2048, 2048), (1536, 2752)])
def test_sigmas_are_the_saved_nodes_at_every_resolution(width, height):
    # The checkpoint turns dynamic shifting off, so no resolution changes the nodes.
    scheduler = _turbo_config(width, height).scheduler
    assert isinstance(scheduler, Qwen21TurboScheduler)
    assert [float(s) for s in scheduler.sigmas] == pytest.approx([*MODEL_INDEX_SIGMAS, 0.0], abs=1e-7)


@pytest.mark.fast
def test_only_the_turbo_model_maps_linear():
    turbo = ModelConfig.qwen_image_21_turbo()
    assert Qwen21TurboScheduler.for_model(turbo, "linear") == "qwen21_turbo"
    assert Qwen21TurboScheduler.for_model(turbo, "flow_match_euler_discrete") == "flow_match_euler_discrete"
    assert Qwen21TurboScheduler.for_model(ModelConfig.qwen_image_21(), "linear") == "linear"


@pytest.mark.fast
def test_timestep_routing_lands_on_sigma_nodes():
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    config = _turbo_config()
    sigmas = [float(s) for s in config.scheduler.sigmas]
    for t in range(8):
        assert float(Qwen21Transformer._compute_timestep(t, config)[0]) == pytest.approx(sigmas[t])


@pytest.mark.fast
def test_euler_step_between_nodes():
    scheduler = _turbo_config().scheduler
    latents = mx.ones((1, 4))
    noise = mx.full((1, 4), -0.5)
    dt = float(scheduler.sigmas[1] - scheduler.sigmas[0])
    out = scheduler.step(noise, 0, latents)
    assert float(out[0][0]) == pytest.approx(1.0 - 0.5 * dt)


@pytest.mark.fast
def test_rejects_other_step_counts():
    config = Config(model_config=ModelConfig.qwen_image_21_turbo(), num_inference_steps=40, width=1024, height=1024)
    with pytest.raises(ValueError, match="8 saved sigma nodes"):
        Qwen21TurboScheduler(config)


def _check(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", *argv])
    parser = module.build_parser()
    args = parser.parse_args()
    model_config = ModelConfig.from_name(args.model or "qwen-image-2.1")
    Qwen21TurboScheduler.check_args(parser, args, model_config)
    return args


@pytest.mark.fast
@pytest.mark.parametrize("module", [qwen21_generate, qwen21_edit_generate])
def test_cli_defaults_pass(monkeypatch, module):
    args = _check(monkeypatch, module, "--model", "qwen-image-2.1-turbo")
    assert args.steps == 8


@pytest.mark.fast
@pytest.mark.parametrize("module", [qwen21_generate, qwen21_edit_generate])
@pytest.mark.parametrize(
    "argv,message",
    [
        (("--steps", "20"), "Use --steps 8"),
        (("--guidance", "4"), "Use --guidance 1"),
        (("--scheduler", "viggle_turbo", "--steps", "8"), "viggle_turbo"),
    ],
)
def test_cli_rejects_settings_turbo_cannot_run(monkeypatch, capsys, module, argv, message):
    with pytest.raises(SystemExit):
        _check(monkeypatch, module, "--model", "qwen-image-2.1-turbo", *argv)
    assert message in capsys.readouterr().err


@pytest.mark.fast
@pytest.mark.parametrize("module", [qwen21_generate, qwen21_edit_generate])
def test_cli_base_model_is_not_checked(monkeypatch, module):
    args = _check(monkeypatch, module, "--model", "qwen-image-2.1", "--steps", "20", "--guidance", "4")
    assert args.steps == 20


@pytest.mark.fast
def test_txt2img_text_encoder_loads_without_a_shard_index(tmp_path):
    # Qwen-Image-2.1-Turbo ships text_encoder/model.safetensors with no model.safetensors.index.json.
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

    component = next(c for c in Qwen21WeightDefinition.get_components() if c.name == "text_encoder")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"a.weight": mx.ones((2, 2))})
    weights = WeightLoader._load_safetensors(tmp_path, component.loading_mode, component.weight_files)
    assert list(weights) == ["a.weight"]


@pytest.mark.fast
@pytest.mark.parametrize("module", [qwen21_generate, qwen21_edit_generate])
def test_cli_local_turbo_folder_defaults_to_eight_steps(monkeypatch, module):
    # The folder name matches the "qwen" alias (20 steps) first. Turbo replaces that default.
    monkeypatch.setattr(
        sys, "argv", ["prog", "--prompt", "test", "--model", "/models/Qwen--Qwen-Image-2.1-Turbo-mflux-q8"]
    )
    parser = module.build_parser()
    args = parser.parse_args()
    Qwen21TurboScheduler.check_args(parser, args, ModelConfig.qwen_image_21_turbo())
    assert args.steps == 8


@pytest.mark.fast
def test_python_api_step_default_follows_the_model():
    import inspect

    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    assert Qwen21TurboScheduler.default_steps(ModelConfig.qwen_image_21_turbo()) == 8
    assert Qwen21TurboScheduler.default_steps(ModelConfig.qwen_image_21()) == 40
    for variant in (QwenImage21, QwenImage21Edit):
        assert inspect.signature(variant.generate_image).parameters["num_inference_steps"].default is None


@pytest.mark.fast
def test_turbo_controlnet_entry_runs_on_the_turbo_schedule():
    controlnet = ModelConfig.from_name("qwen-2.1-turbo-controlnet")
    assert controlnet is ModelConfig.qwen_image_21_turbo_controlnet_union()
    assert controlnet.model_name == "Qwen/Qwen-Image-2.1-Turbo"
    assert controlnet.controlnet_model == "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union"
    assert Qwen21TurboScheduler.is_turbo(controlnet)
    assert Qwen21TurboScheduler.for_model(controlnet, "linear") == "qwen21_turbo"
    assert Qwen21TurboScheduler.default_steps(controlnet) == 8
    assert model_inference_steps("qwen-image-2.1-turbo-controlnet-union") == 8
    base_controlnet = ModelConfig.qwen_image_21_controlnet_union()
    assert not Qwen21TurboScheduler.is_turbo(base_controlnet)
    assert Qwen21TurboScheduler.default_steps(base_controlnet) == 40


@pytest.mark.fast
def test_controlnet_cli_checks_the_turbo_entry(monkeypatch, capsys):
    args = _check(monkeypatch, qwen21_controlnet_generate, "--model", "qwen-image-2.1-turbo-controlnet")
    assert args.steps == 8
    with pytest.raises(SystemExit):
        _check(monkeypatch, qwen21_controlnet_generate, "--model", "qwen-image-2.1-turbo-controlnet", "--steps", "20")
    assert "Use --steps 8" in capsys.readouterr().err


@pytest.mark.fast
def test_controlnet_python_api_step_default_is_none():
    import inspect

    from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet

    assert inspect.signature(QwenImage21Controlnet.generate_image).parameters["num_inference_steps"].default is None
