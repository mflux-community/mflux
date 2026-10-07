import inspect
import sys

import PIL.Image
import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.qwen.cli import qwen_image_generate as cli
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


class FakeQwenImage(FakeModel):
    real = QwenImage


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeQwenImage.instances.clear()
    monkeypatch.setattr(cli, "QwenImage", FakeQwenImage)
    reset_cli_globals(monkeypatch)
    yield
    FakeQwenImage.instances.clear()


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
    # Every value differs from the parser default so a hardcoded default goes red.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "2x", "--height", "3x",
        "--image", str(ref_png), "0.55",
        "--guidance", "2.5",
        "--negative-prompt", "blurry",
        "--scheduler", "flow_match_euler_discrete",
        "--pid-decode", "--pid-degrade-sigma", "0.2",
        "--lora", str(lora_file), "0.5",
        "--no-bake-lora",
        "-q", "8",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen", *argv])
    cli.main()
    assert len(FakeQwenImage.instances) == 1
    return FakeQwenImage.instances[0]


def expected_generate_call(seed, ref_png):
    # Literals from the CLI contract, not read back from the code: 2x wide and 3x high on a
    # 64x32 reference is 128x96.
    return {
        "seed": seed,
        "prompt": "a puffin",
        "negative_prompt": "blurry",
        "width": 128,
        "height": 96,
        "guidance": 2.5,
        "scheduler": "flow_match_euler_discrete",
        "image_path": ref_png,
        "num_inference_steps": 3,
        "image_strength": 0.55,
        "pid_decode": True,
        "pid_degrade_sigma": 0.2,
    }


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, ref_png, lora_file, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.bound_init() == {
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": False,
        "model_config": AVAILABLE_MODELS["qwen-image"],
    }
    assert model.generate_calls == [expected_generate_call(7, ref_png), expected_generate_call(8, ref_png)]
    # With more than one seed parse_args renames --output to <stem>_seed_{seed}; that
    # parse-time rename is part of the contract pinned here.
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_uses_guidance_3_5_when_the_flag_is_absent_and_keeps_an_explicit_zero(monkeypatch):
    # --help: "Default varies by tool: 3.5 for most". An explicit 0 is a value, not "absent".
    assert run_main(monkeypatch, ["--prompt", "x"]).generate_calls[0]["guidance"] == 3.5
    FakeQwenImage.instances.clear()
    assert run_main(monkeypatch, ["--prompt", "x", "--guidance", "0"]).generate_calls[0]["guidance"] == 0.0


@pytest.mark.fast
def test_main_keeps_the_default_flags_off(monkeypatch, tmp_path):
    # full_argv sets every flag to a non-default value, so a flag hardcoded on would pass there.
    model = run_main(monkeypatch, ["--prompt", "x", "--output", str(tmp_path / "out.png")])
    call = model.generate_calls[0]
    assert (call["negative_prompt"], call["image_path"], call["pid_decode"]) == ("", None, False)
    assert (call["width"], call["height"]) == (1024, 1024)
    # The remaining defaults are the parser's and the command's: full_argv passes every one of
    # them, so a value hardcoded to what it passes would stay green there.
    assert model.bound_init()["quantize"] is None
    assert model.bound_init()["bake_lora"] is True
    assert (model.bound_init()["lora_paths"], model.bound_init()["lora_scales"]) == (None, None)
    assert call["guidance"] == 3.5
    assert call["scheduler"] == "linear"
    assert call["num_inference_steps"] == ui_defaults.MODEL_INFERENCE_STEPS["qwen-image"]
    assert call["image_strength"] == ui_defaults.IMAGE_STRENGTH
    assert call["pid_degrade_sigma"] == 0.0
    assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "dev"],
        ["--model", "qwen-image-edit"],
        ["--model", "someone/qwen-mirror", "--base-model", "dev"],
    ],
    ids=["foreign-built-in-name", "edit-sibling-name", "repo-id-with-foreign-base"],
)
def test_main_rejects_a_model_name_this_command_cannot_run_before_building_anything(monkeypatch, argv):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen", "--prompt", "x", *argv])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeQwenImage.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv", "model_path"),
    [
        (["--model", "qwen"], None),
        (["--model", "qwen-2512"], None),
        (["--model", "someone/my-finetune", "--base-model", "qwen-image"], "someone/my-finetune"),
    ],
    ids=["alias", "2512-alias", "repo-id-with-own-base"],
)
def test_main_builds_the_qwen_image_config_for_its_own_names(monkeypatch, argv, model_path):
    model = run_main(monkeypatch, ["--prompt", "x", *argv])
    assert model.bound_init()["model_config"] is AVAILABLE_MODELS["qwen-image"]
    assert model.bound_init()["model_path"] == model_path


@pytest.mark.fast
def test_main_hands_the_callbacks_the_size_flags_as_passed(monkeypatch, ref_png, full_argv):
    # main() resolves "2x"/"3x" once for the run, but the args a callback is given keep the
    # flags as typed, the way they were before the command steps existed.
    handed = {}
    register = cli.CallbackManager.register_callbacks

    def keep_args(**kwargs):
        handed["args"] = kwargs["args"]
        return register(**kwargs)

    monkeypatch.setattr(cli.CallbackManager, "register_callbacks", keep_args)
    model = run_main(monkeypatch, full_argv)
    assert (str(handed["args"].width), str(handed["args"].height)) == ("2x", "3x")
    assert [(c["width"], c["height"]) for c in model.generate_calls] == [(128, 96), (128, 96)]


@pytest.mark.fast
def test_main_passes_a_repo_id_as_model_path(monkeypatch):
    model = run_main(monkeypatch, ["--prompt", "x", "--model", "someone/qwen-mirror"])
    assert model.bound_init()["model_path"] == "someone/qwen-mirror"


@pytest.mark.fast
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, capsys, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_wires_the_stepwise_handler_with_the_qwen_latent_creator(monkeypatch, tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator

    model = run_main(monkeypatch, ["--prompt", "x", "--stepwise-image-output-dir", str(tmp_path / "steps")])
    handlers = [cb for cb in model.callbacks.in_loop if isinstance(cb, StepwiseHandler)]
    assert len(handlers) == 1
    assert handlers[0].latent_creator is QwenLatentCreator


@pytest.mark.fast
def test_main_prints_a_cancellation_and_still_reports_memory(monkeypatch, capsys, full_argv):
    from mflux.utils.exceptions import StopImageGenerationException

    def cancel_first(self, **kwargs):
        self.bind_generate(**kwargs)
        self.generate_calls.append(kwargs)
        raise StopImageGenerationException("Stopping image generation at step 1/3")

    monkeypatch.setattr(FakeQwenImage, "generate_image", cancel_first)
    model = run_main(monkeypatch, full_argv)
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
    assert "Peak MLX memory: " in out
    assert len(model.generate_calls) == 1  # a cancel ends the run; seed 8 never starts


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

    monkeypatch.setattr(FakeQwenImage, "generate_image", boom)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen", *full_argv])
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

    monkeypatch.setattr(FakeQwenImage, "generate_image", rewrite_after_first)
    run_main(
        monkeypatch,
        ["--prompt-file", str(prompt_file), "--seed", "7", "8", "--output", str(tmp_path / "out.png")],
    )
    assert seen == ["a", "b"]


@pytest.mark.fast
def test_main_reports_a_missing_reference_image_before_it_reads_the_prompt(monkeypatch, capsys, tmp_path):
    # Sizes left at "auto" follow the reference image, and main() resolves them before the
    # seed loop, so a missing image stops the run before a missing prompt file is noticed.
    # The memory report still prints, because the size is resolved inside the try.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen",
            "--prompt-file",
            str(tmp_path / "missing.txt"),
            "--image",
            str(tmp_path / "missing.png"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        cli.main()
    assert FakeQwenImage.instances[0].generate_calls == []
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_reads_the_reference_image_size_once_for_all_seeds(monkeypatch, ref_png):
    # The size comes from the image header once per run; later seeds do not reopen the file.
    def vanish_after_first(self, **kwargs):
        image = FakeModel.generate_image(self, **kwargs)
        if ref_png.exists():
            ref_png.rename(ref_png.with_suffix(".gone"))
        return image

    monkeypatch.setattr(FakeQwenImage, "generate_image", vanish_after_first)
    model = run_main(monkeypatch, ["--prompt", "x", "--seed", "1", "2", "--image", str(ref_png)])
    assert [(c["width"], c["height"]) for c in model.generate_calls] == [(64, 32), (64, 32)]


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_returns_the_config_the_model_class_defaults_to(monkeypatch, tmp_path):
    # load() passes this config explicitly; that builds the same model as passing none only
    # while it is the very object QwenImage defaults to. Builds nothing, opens no file.
    args = args_for(
        monkeypatch,
        ["--prompt-file", str(tmp_path / "missing.txt"), "--image", str(tmp_path / "missing.png")],
    )
    config = cli.QwenImageCommand.validate(args)
    assert config is AVAILABLE_MODELS["qwen-image"]
    assert config is inspect.signature(QwenImage.__init__).parameters["model_config"].default
    assert FakeQwenImage.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv",
    [["--model", "dev"], ["--model", "someone/qwen-mirror", "--base-model", "dev"]],
    ids=["foreign-built-in-name", "repo-id-with-foreign-base"],
)
def test_validate_rejects_a_model_name_this_command_cannot_run(monkeypatch, argv):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", *argv])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.QwenImageCommand.validate(args)


@pytest.mark.fast
def test_load_returns_a_bare_model_on_the_validated_config(monkeypatch, lora_file):
    args = args_for(monkeypatch, ["--prompt", "x", "--lora", str(lora_file), "0.5", "-q", "8"])
    model = cli.QwenImageCommand.load(args)
    assert model.bound_init() == {
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": True,
        "model_config": cli.QwenImageCommand.validate(args),
    }
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt_and_resolved_dims(monkeypatch, ref_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.QwenImageCommand.load(args)
    image = cli.QwenImageCommand.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [{**expected_generate_call(7, ref_png), "prompt": "a different puffin"}]
    assert (str(args.width), str(args.height)) == ("2x", "3x")


@pytest.mark.fast
def test_generate_applies_the_guidance_default_without_writing_it_into_args(monkeypatch):
    # A UI calls generate() without main(); it must still get the command's 3.5 (a None
    # would reach the model's Config as 0.0), and its args must come back as it passed them.
    args = args_for(monkeypatch, ["--prompt", "x"])
    model = cli.QwenImageCommand.load(args)
    cli.QwenImageCommand.generate(model, args, 1, "x")
    assert model.generate_calls[0]["guidance"] == 3.5
    assert args.guidance is None
