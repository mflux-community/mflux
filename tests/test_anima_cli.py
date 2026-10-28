import copy
import json
import sys
import warnings

import mlx.core as mx
import pytest

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.cli.capabilities import describe_command
from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.anima.cli import anima_generate
from mflux.models.anima.variants.txt2img.anima import Anima
from mflux.models.common.cli import save
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution


class _StubAnima:
    # Records what the CLI hands the model instead of loading the weights.
    instances: list = []
    DEFAULT_GUIDANCE = Anima.DEFAULT_GUIDANCE

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generated: list = []
        self.callbacks = CallbackRegistry()
        _StubAnima.instances.append(self)

    def guidance_or_default(self, guidance):
        return Anima.DEFAULT_GUIDANCE[self.kwargs["model_config"].aliases[0]] if guidance is None else guidance

    def generate_image(self, **kwargs):
        self.generated.append(kwargs)
        return type("Image", (), {"save": staticmethod(lambda path, export_json_metadata=False: None)})()


@pytest.fixture
def run_cli(monkeypatch):
    def run(*argv):
        _StubAnima.instances.clear()
        monkeypatch.setattr(sys, "argv", ["mflux-generate-anima", "--prompt", "1girl, solo", "--seed", "1", *argv])
        monkeypatch.setattr(anima_generate, "Anima", _StubAnima)
        monkeypatch.setattr(anima_generate.CallbackManager, "register_callbacks", staticmethod(lambda **kw: None))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            anima_generate.main()
        return _StubAnima.instances[0], [str(w.message) for w in caught]

    return run


@pytest.mark.fast
class TestAnimaGenerateCli:
    def test_the_default_is_turbo_at_1024_with_ten_steps_and_er_sde(self, run_cli):
        model, warned = run_cli()

        assert model.kwargs["model_config"] is AVAILABLE_MODELS["anima-turbo"]
        call = model.generated[0]
        assert (call["width"], call["height"], call["num_inference_steps"]) == (1024, 1024, 10)
        assert call["guidance"] is None
        assert call["scheduler"] == "er_sde"
        assert warned == []

    @pytest.mark.parametrize(("name", "key", "steps"), [("anima", "anima-turbo", 10), ("anima-aesthetic", "anima-aesthetic", 30), ("anima-base", "anima-base", 30)])  # fmt: off
    def test_each_variant_is_selected_by_name_with_its_step_count(self, run_cli, name, key, steps):
        model, _ = run_cli("--model", name)

        assert model.kwargs["model_config"] is AVAILABLE_MODELS[key]
        assert model.generated[0]["num_inference_steps"] == steps

    def test_steps_guidance_scheduler_and_negative_prompt_reach_the_model(self, run_cli):
        model, warned = run_cli(*"--model anima-aesthetic --steps 40 --guidance 5 --scheduler euler --negative-prompt blurry".split())  # fmt: off

        call = model.generated[0]
        assert (call["num_inference_steps"], call["guidance"], call["scheduler"]) == (40, 5.0, "euler")
        assert call["negative_prompt"] == "blurry"
        assert warned == []

    def test_a_negative_prompt_at_guidance_one_warns_that_it_is_ignored(self, run_cli):
        _, warned = run_cli("--negative-prompt", "blurry")

        assert any(message.startswith("--negative-prompt is ignored at guidance 1.0") for message in warned)

    def test_an_unknown_scheduler_fails_before_the_model_loads(self, run_cli):
        with pytest.raises(ValueError, match="er_sde, euler"):
            run_cli("--scheduler", "dpmpp_2m")
        assert _StubAnima.instances == []

    @pytest.mark.parametrize("steps", ["0", "1001"])
    def test_a_step_count_outside_the_schedule_fails_before_the_model_loads(self, run_cli, steps):
        with pytest.raises(ValueError, match="1 to 1000 steps"):
            run_cli("--steps", steps)
        assert _StubAnima.instances == []

    def test_a_scaled_size_is_refused_since_there_is_no_image_to_scale(self, run_cli):
        with pytest.raises(SystemExit):
            run_cli("--width", "2x")

    def test_a_size_that_is_not_a_multiple_of_16_fails_before_the_model_loads(self, run_cli):
        with pytest.raises(ValueError, match="multiples of 16"):
            run_cli("--width", "1000")
        assert _StubAnima.instances == []

    def test_quantize_lora_and_a_saved_model_reach_the_model(self, run_cli, tmp_path):
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"")
        model, _ = run_cli("--model", "/models/Anima-aesthetic-mflux-q8", "-q", "8", "--lora", str(lora))

        assert model.kwargs["model_path"] == "/models/Anima-aesthetic-mflux-q8"
        assert model.kwargs["quantize"] == 8
        assert model.kwargs["lora_paths"] == [str(lora)]
        assert model.kwargs["model_config"] is AVAILABLE_MODELS["anima-aesthetic"]

    def test_float16_compute_reaches_the_model_only_when_asked(self, run_cli):
        assert run_cli()[0].kwargs["compute_precision"] is None
        assert run_cli("--compute-precision", "float16")[0].kwargs["compute_precision"] == mx.float16

    def test_a_foreign_model_is_rejected(self, run_cli):
        with pytest.raises(Exception, match="only accepts the aliases"):
            run_cli("--model", "krea-2")

    def test_capabilities_report_the_negative_prompt_as_conditional(self):
        entry = describe_command("mflux-generate-anima", anima_generate.__name__)
        status = {option["flag"]: option["status"] for option in entry["options"]}

        assert status["--negative-prompt"] == "conditional"
        assert "--lora" in status


@pytest.mark.fast
class TestAnimaResolution:
    @pytest.mark.parametrize(
        ("directory", "expected"),
        [
            ("~/models/Anima-turbo-mflux-q8", "anima-turbo"),
            ("~/models/Anima-aesthetic-mflux-q4", "anima-aesthetic"),
            ("~/models/Anima-base-mflux-q8", "anima-base"),
            ("~/models/anima-q8", "anima-turbo"),
            ("~/models/my-finetune", "anima-turbo"),
        ],
    )
    def test_a_saved_directory_resolves_to_the_variant_its_name_selects(self, directory, expected):
        config = ConfigResolution.resolve_restricted(
            None, anima_generate.DEFAULT_MODEL, model_path=directory, extra_keys=anima_generate.FAMILY_MODELS
        )

        assert config is AVAILABLE_MODELS[expected]
        if "anima" in directory.lower():
            assert ui_defaults.model_inference_steps(directory) == ui_defaults.MODEL_INFERENCE_STEPS[expected]

    def test_the_shared_repo_id_runs_turbo_unless_a_base_model_names_the_variant(self):
        repo = "circlestone-labs/Anima"
        family = dict(extra_keys=anima_generate.FAMILY_MODELS)

        assert ConfigResolution.resolve(repo) is AVAILABLE_MODELS["anima-turbo"]
        assert ConfigResolution.resolve_restricted(repo, "anima-turbo", model_path=repo, **family) is AVAILABLE_MODELS["anima-turbo"]  # fmt: off
        assert ConfigResolution.resolve_restricted(repo, "anima-turbo", model_path=repo, base_model="anima-base", **family) is AVAILABLE_MODELS["anima-base"]  # fmt: off

    def test_a_sidecar_names_the_variant_of_an_aesthetic_or_base_run(self):
        # What generate_image hands the metadata: the shared repo id, and the variant as base_model.
        for key in ("anima-aesthetic", "anima-base"):
            stub = Anima.__new__(Anima)
            object.__setattr__(stub, "model_config", copy.copy(AVAILABLE_MODELS[key]))
            recorded = Anima._metadata_model_config(stub)
            assert (recorded.model_name, recorded.base_model) == ("circlestone-labs/Anima", key)
        stub = Anima.__new__(Anima)
        object.__setattr__(stub, "model_config", AVAILABLE_MODELS["anima-turbo"])
        assert Anima._metadata_model_config(stub).base_model is None

    def test_every_variant_saves_through_anima(self):
        for key in ("anima-turbo", "anima-aesthetic", "anima-base"):
            assert save.MODEL_CLASSES[key] is Anima

    def test_the_card_defaults(self):
        assert Anima.DEFAULT_GUIDANCE == {"anima-turbo": 1.0, "anima-aesthetic": 4.0, "anima-base": 4.0}
        assert {key: ui_defaults.MODEL_INFERENCE_STEPS[key] for key in Anima.DEFAULT_STEPS} == Anima.DEFAULT_STEPS


@pytest.mark.fast
class TestAnimaReplay:
    @pytest.mark.parametrize(("recorded", "expected"), [({"sampler": "euler"}, "euler"), ({}, "er_sde")])
    def test_a_sidecar_replays_its_sampler(self, run_cli, tmp_path, recorded, expected):
        # --make-conf records the sampler; a sidecar from 0.24.0 has none and runs the default
        sidecar = tmp_path / "image.metadata.json"
        sidecar.write_text(json.dumps({"prompt": "1girl", "seed": 1, **recorded}))
        model, _ = run_cli("-C", str(sidecar))
        assert model.generated[0]["scheduler"] == expected

    def test_the_flag_overrides_the_sidecar_sampler(self, run_cli, tmp_path):
        sidecar = tmp_path / "image.metadata.json"
        sidecar.write_text(json.dumps({"prompt": "1girl", "seed": 1, "sampler": "euler"}))
        model, _ = run_cli("-C", str(sidecar), "--scheduler", "er_sde")
        assert model.generated[0]["scheduler"] == "er_sde"


@pytest.mark.fast
class TestAnimaReplayGuards:
    def test_a_sidecar_sampler_outside_anima_names_is_refused_by_the_parser(self, run_cli, tmp_path):
        # A dotted name would otherwise reach the scheduler lookup, which imports it
        sidecar = tmp_path / "image.metadata.json"
        sidecar.write_text(json.dumps({"prompt": "1girl", "seed": 1, "sampler": "os.path.join"}))
        with pytest.raises(SystemExit):
            run_cli("-C", str(sidecar))
        assert _StubAnima.instances == []

    def test_a_weight_that_is_not_finite_fails_before_the_model_loads(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-anima", "--prompt", "(cat:nan)", "--seed", "1"])
        monkeypatch.setattr(anima_generate, "Anima", _StubAnima)
        _StubAnima.instances.clear()
        with pytest.raises(ValueError, match="not a finite number"):
            anima_generate.main()
        assert _StubAnima.instances == []

    def test_a_negative_prompt_weight_that_is_not_finite_fails_before_the_model_loads(self, monkeypatch):
        argv = [
            "mflux-generate-anima",
            "--prompt",
            "cat",
            "--negative-prompt",
            "(x:nan)",
            "--guidance",
            "4",
            "--seed",
            "1",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(anima_generate, "Anima", _StubAnima)
        _StubAnima.instances.clear()
        with pytest.raises(ValueError, match="not a finite number"):
            anima_generate.main()
        assert _StubAnima.instances == []

    def test_a_negative_prompt_is_not_checked_at_guidance_1(self, monkeypatch):
        argv = ["mflux-generate-anima", "--prompt", "cat", "--negative-prompt", "(x:nan)", "--seed", "1"]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(anima_generate, "Anima", _StubAnima)
        _StubAnima.instances.clear()
        anima_generate.AnimaCommand.validate(anima_generate.build_parser().parse_args())

    @pytest.mark.parametrize("flag", ["--prompt", "--negative-prompt"])
    def test_a_weight_out_of_range_fails_before_the_model_loads(self, monkeypatch, flag):
        other = [] if flag == "--prompt" else ["--prompt", "cat", "--guidance", "4"]
        argv = ["mflux-generate-anima", "--seed", "1", flag, "(cat:1e39)", *other]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(anima_generate, "Anima", _StubAnima)
        _StubAnima.instances.clear()
        with pytest.raises(ValueError, match="float32 range"):
            anima_generate.main()
        assert _StubAnima.instances == []

    def test_a_prompt_file_is_checked_before_the_model_loads(self, monkeypatch, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("(cat:nan)")
        monkeypatch.setattr(sys, "argv", ["mflux-generate-anima", "--prompt-file", str(prompt_file), "--seed", "1"])
        monkeypatch.setattr(anima_generate, "Anima", _StubAnima)
        _StubAnima.instances.clear()
        with pytest.raises(ValueError, match="not a finite number"):
            anima_generate.main()
        assert _StubAnima.instances == []


@pytest.mark.fast
def test_an_anima_sidecar_keeps_another_commands_scheduler(monkeypatch, tmp_path):
    # Only Anima's --scheduler takes er_sde or euler; mflux-generate would fail after the model loaded
    from mflux.models.flux.cli import flux_generate

    sidecar = tmp_path / "image.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "1girl", "seed": 1, "sampler": "euler"}))
    monkeypatch.setattr(sys, "argv", ["mflux-generate", "--model", "schnell", "-C", str(sidecar)])
    assert flux_generate.build_parser().parse_args().scheduler != "euler"
