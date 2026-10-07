import inspect
import sys

import PIL.Image
import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.qwen.cli import qwen_image_edit_generate as cli
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals


class FakeQwenImageEdit(FakeModel):
    real = QwenImageEdit


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeQwenImageEdit.instances.clear()
    monkeypatch.setattr(cli, "QwenImageEdit", FakeQwenImageEdit)
    reset_cli_globals(monkeypatch)
    yield
    FakeQwenImageEdit.instances.clear()


@pytest.fixture
def first_png(tmp_path):
    path = tmp_path / "first.png"
    PIL.Image.new("RGB", (64, 32)).save(path)
    return path


@pytest.fixture
def second_png(tmp_path):
    # A different size from the first image, so sizing from the wrong image goes red.
    path = tmp_path / "second.png"
    PIL.Image.new("RGB", (48, 80)).save(path)
    return path


@pytest.fixture
def lora_file(tmp_path):
    path = tmp_path / "style.safetensors"
    path.write_bytes(b"")
    return path


@pytest.fixture
def full_argv(tmp_path, first_png, second_png, lora_file):
    # Every value differs from the parser default so a hardcoded default goes red.
    return [
        "--prompt", "a puffin",
        "--seed", "7", "8",
        "--steps", "3",
        "--width", "2x", "--height", "3x",
        "--image-paths", str(first_png), str(second_png),
        "--guidance", "3.0",
        "--negative-prompt", "blurry",
        "--scheduler", "flow_match_euler_discrete",
        "--lora", str(lora_file), "0.5",
        "--no-bake-lora",
        "-q", "8",
        "--make-conf",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def args_for(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen-edit", *argv])
    return cli.build_parser().parse_args()


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen-edit", *argv])
    cli.main()
    assert len(FakeQwenImageEdit.instances) == 1
    return FakeQwenImageEdit.instances[0]


def expected_generate_call(seed, first_png, second_png):
    # 2x wide and 3x high on the first image (64x32) is 128x96. The paths reach the model as
    # strings, the first one also as image_path.
    return {
        "seed": seed,
        "prompt": "a puffin",
        "negative_prompt": "blurry",
        "width": 128,
        "height": 96,
        "guidance": 3.0,
        "image_path": str(first_png),
        "image_paths": [str(first_png), str(second_png)],
        "num_inference_steps": 3,
        "scheduler": "flow_match_euler_discrete",
    }


@pytest.mark.fast
def test_main_call_sequence_is_pinned(monkeypatch, tmp_path, first_png, second_png, lora_file, full_argv):
    model = run_main(monkeypatch, full_argv)
    assert model.bound_init() == {
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": False,
        "model_config": AVAILABLE_MODELS["qwen-image-edit"],
    }
    assert model.generate_calls == [
        expected_generate_call(7, first_png, second_png),
        expected_generate_call(8, first_png, second_png),
    ]
    # With more than one seed parse_args renames --output to <stem>_seed_{seed}; that
    # parse-time rename is part of the contract pinned here.
    assert [img.saves for img in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), True)],
        [(str(tmp_path / "out_seed_8.png"), True)],
    ]


@pytest.mark.fast
def test_main_uses_guidance_2_5_when_the_flag_is_absent_and_keeps_an_explicit_zero(monkeypatch, first_png):
    # The edit command's own default is 2.5, the value every README edit example passes.
    model = run_main(monkeypatch, ["--prompt", "x", "--image-paths", str(first_png)])
    assert model.generate_calls[0]["guidance"] == 2.5
    FakeQwenImageEdit.instances.clear()
    model = run_main(monkeypatch, ["--prompt", "x", "--image-paths", str(first_png), "--guidance", "0"])
    assert model.generate_calls[0]["guidance"] == 0.0


@pytest.mark.fast
def test_main_scales_from_the_first_image(monkeypatch, first_png, second_png):
    model = run_main(
        monkeypatch,
        ["--prompt", "x", "--image-paths", str(first_png), str(second_png), "--width", "2x", "--height", "320"],
    )
    assert (model.generate_calls[0]["width"], model.generate_calls[0]["height"]) == (128, 320)


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv",
    [["--model", "dev"], ["--model", "qwen-image"], ["--model", "someone/edit-mirror", "--base-model", "dev"]],
    ids=["foreign-built-in-name", "txt2img-sibling-name", "repo-id-with-foreign-base"],
)
def test_main_rejects_a_model_name_this_command_cannot_run_before_building_anything(monkeypatch, first_png, argv):
    from mflux.utils.exceptions import ModelConfigError

    monkeypatch.setattr(
        sys, "argv", ["mflux-generate-qwen-edit", "--prompt", "x", "--image-paths", str(first_png), *argv]
    )
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.main()
    assert FakeQwenImageEdit.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    ("argv", "model_path"),
    [
        (["--model", "qwen-edit"], None),
        (["--model", "qwen-edit-2511"], None),
        (["--model", "someone/my-finetune", "--base-model", "qwen-image-edit"], "someone/my-finetune"),
    ],
    ids=["alias", "2511-alias", "repo-id-with-own-base"],
)
def test_main_builds_the_edit_config_for_its_own_names(monkeypatch, first_png, argv, model_path):
    model = run_main(monkeypatch, ["--prompt", "x", "--image-paths", str(first_png), *argv])
    assert model.bound_init()["model_config"] is AVAILABLE_MODELS["qwen-image-edit"]
    assert model.bound_init()["model_path"] == model_path


@pytest.mark.fast
def test_main_hands_the_callbacks_the_size_flags_as_passed(monkeypatch, first_png, second_png, full_argv):
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
def test_main_registers_the_cli_callbacks_after_load_and_before_generate(monkeypatch, capsys, full_argv):
    from mflux.callbacks.instances.battery_saver import BatterySaver
    from mflux.callbacks.instances.memory_saver import MemorySaver

    model = run_main(monkeypatch, full_argv)
    assert [type(cb) for cb in model.callbacks.before_loop] == [BatterySaver, MemorySaver]
    assert model.before_loop_count_at_generate == [2, 2]
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_wires_the_stepwise_handler_with_the_qwen_latent_creator(monkeypatch, tmp_path, first_png):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator

    model = run_main(
        monkeypatch,
        ["--prompt", "x", "--image-paths", str(first_png), "--stepwise-image-output-dir", str(tmp_path / "steps")],
    )
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

    monkeypatch.setattr(FakeQwenImageEdit, "generate_image", cancel_first)
    model = run_main(monkeypatch, full_argv)
    out = capsys.readouterr().out
    assert "Stopping image generation at step 1/3" in out
    assert "Peak MLX memory: " in out
    assert len(model.generate_calls) == 1


@pytest.mark.fast
def test_main_prints_a_missing_prompt_file_and_generates_nothing(monkeypatch, capsys, tmp_path, first_png):
    model = run_main(monkeypatch, ["--prompt-file", str(tmp_path / "missing.txt"), "--image-paths", str(first_png)])
    assert "Prompt file does not exist" in capsys.readouterr().out
    assert model.generate_calls == []


@pytest.mark.fast
def test_main_lets_a_missing_input_image_escape_after_loading_and_still_reports_memory(monkeypatch, capsys, tmp_path):
    # A typed --image-paths is not checked at parse time; the size lookup opens it, and that
    # error keeps its traceback.
    monkeypatch.setattr(
        sys, "argv", ["mflux-generate-qwen-edit", "--prompt", "x", "--image-paths", str(tmp_path / "gone.png")]
    )
    with pytest.raises(FileNotFoundError):
        cli.main()
    assert FakeQwenImageEdit.instances[0].generate_calls == []
    assert "Peak MLX memory: " in capsys.readouterr().out


@pytest.mark.fast
def test_main_keeps_the_default_flags_off(monkeypatch, tmp_path, first_png):
    # full_argv sets every flag to a non-default value, so a flag hardcoded on would pass there.
    out = tmp_path / "out.png"
    model = run_main(monkeypatch, ["--prompt", "x", "--image-paths", str(first_png), "--output", str(out)])
    call = model.generate_calls[0]
    assert call["negative_prompt"] == ""
    assert (call["width"], call["height"]) == (64, 32)  # "auto" keeps the first image's size
    assert model.images[0].saves == [(str(out), False)]
    # The remaining defaults are the parser's and the command's: full_argv passes every one of
    # them, so a value hardcoded to what it passes would stay green there.
    assert model.bound_init()["quantize"] is None
    assert model.bound_init()["bake_lora"] is True
    assert (model.bound_init()["lora_paths"], model.bound_init()["lora_scales"]) == (None, None)
    assert call["guidance"] == 2.5
    assert call["scheduler"] == "linear"
    assert call["num_inference_steps"] == ui_defaults.MODEL_INFERENCE_STEPS["qwen-image-edit"]


@pytest.mark.fast
def test_main_passes_a_repo_id_and_the_load_flags_to_the_constructor(monkeypatch, first_png, lora_file):
    model = run_main(
        monkeypatch,
        [
            *["--prompt", "x", "--image-paths", str(first_png)],
            *["--model", "someone/edit-mirror", "-q", "8", "--lora", str(lora_file), "0.5"],
        ],
    )
    assert model.bound_init() == {
        "quantize": 8,
        "model_path": "someone/edit-mirror",
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": True,
        "model_config": AVAILABLE_MODELS["qwen-image-edit"],
    }


@pytest.mark.fast
def test_main_reads_the_prompt_file_again_for_each_seed(monkeypatch, tmp_path, first_png):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a")
    seen = []

    def rewrite_after_first(self, **kwargs):
        self.bind_generate(**kwargs)
        seen.append(kwargs["prompt"])
        prompt_file.write_text("b")
        return FakeModel.generate_image(self, **kwargs)

    monkeypatch.setattr(FakeQwenImageEdit, "generate_image", rewrite_after_first)
    run_main(
        monkeypatch,
        [
            "--prompt-file",
            str(prompt_file),
            "--seed",
            "7",
            "8",
            "--image-paths",
            str(first_png),
            "--output",
            str(tmp_path / "out.png"),
        ],
    )
    assert seen == ["a", "b"]


@pytest.mark.fast
def test_main_reports_a_missing_reference_image_before_it_reads_the_prompt(monkeypatch, tmp_path):
    # Sizes left at "auto" follow the first image, and main() resolves them before the seed
    # loop, so a missing image stops the run before a missing prompt file is noticed.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-qwen-edit",
            "--prompt-file",
            str(tmp_path / "missing.txt"),
            "--image-paths",
            str(tmp_path / "missing.png"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        cli.main()
    assert FakeQwenImageEdit.instances[0].generate_calls == []


@pytest.mark.fast
def test_main_reads_the_reference_image_size_once_for_all_seeds(monkeypatch, first_png):
    # The size comes from the image header once per run; later seeds do not reopen the file.
    def vanish_after_first(self, **kwargs):
        image = FakeModel.generate_image(self, **kwargs)
        if first_png.exists():
            first_png.rename(first_png.with_suffix(".gone"))
        return image

    monkeypatch.setattr(FakeQwenImageEdit, "generate_image", vanish_after_first)
    model = run_main(monkeypatch, ["--prompt", "x", "--seed", "1", "2", "--image-paths", str(first_png)])
    assert [(c["width"], c["height"]) for c in model.generate_calls] == [(64, 32), (64, 32)]


# --- the steps a Python caller uses: validate, load, generate ---


@pytest.mark.fast
def test_validate_returns_the_config_the_model_class_defaults_to(monkeypatch, tmp_path):
    # Builds nothing and opens no file: the input images do not exist.
    args = args_for(
        monkeypatch,
        ["--prompt-file", str(tmp_path / "missing.txt"), "--image-paths", str(tmp_path / "missing.png")],
    )
    config = cli.QwenImageEditCommand.validate(args)
    assert config is AVAILABLE_MODELS["qwen-image-edit"]
    assert config is inspect.signature(QwenImageEdit.__init__).parameters["model_config"].default
    assert FakeQwenImageEdit.instances == []


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv",
    [["--model", "qwen-image"], ["--model", "someone/edit-mirror", "--base-model", "dev"]],
    ids=["sibling-built-in-name", "repo-id-with-foreign-base"],
)
def test_validate_rejects_a_model_name_this_command_cannot_run(monkeypatch, first_png, argv):
    from mflux.utils.exceptions import ModelConfigError

    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(first_png), *argv])
    with pytest.raises(ModelConfigError, match="only accepts the aliases"):
        cli.QwenImageEditCommand.validate(args)


@pytest.mark.fast
def test_load_returns_a_bare_model_on_the_validated_config(monkeypatch, first_png, lora_file):
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(first_png), "--lora", str(lora_file), "0.5"])
    model = cli.QwenImageEditCommand.load(args)
    assert model.bound_init()["model_config"] is cli.QwenImageEditCommand.validate(args)
    assert model.bound_init()["lora_paths"] == [str(lora_file)]
    registry = model.callbacks
    assert (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt) == ([], [], [], [])


@pytest.mark.fast
def test_generate_returns_the_image_unsaved_with_the_given_prompt(monkeypatch, first_png, second_png, full_argv):
    args = args_for(monkeypatch, full_argv)
    model = cli.QwenImageEditCommand.load(args)
    image = cli.QwenImageEditCommand.generate(model, args, 7, "a different puffin")
    assert image is model.images[0]
    assert image.saves == []
    assert model.generate_calls == [
        {**expected_generate_call(7, first_png, second_png), "prompt": "a different puffin"}
    ]
    assert (str(args.width), str(args.height)) == ("2x", "3x")


@pytest.mark.fast
@pytest.mark.parametrize("image_paths", [[], None], ids=["empty", "none"])
def test_generate_rejects_args_without_an_input_image(monkeypatch, first_png, image_paths):
    # The parser requires --image-paths; a Python caller building args by hand gets a
    # ValueError that names the field, not an IndexError from inside generate().
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(first_png)])
    model = cli.QwenImageEditCommand.load(args)
    args.image_paths = image_paths
    with pytest.raises(ValueError, match="image_paths"):
        cli.QwenImageEditCommand.generate(model, args, 1, "x")


@pytest.mark.fast
def test_generate_applies_the_guidance_default_without_writing_it_into_args(monkeypatch, first_png):
    args = args_for(monkeypatch, ["--prompt", "x", "--image-paths", str(first_png)])
    model = cli.QwenImageEditCommand.load(args)
    cli.QwenImageEditCommand.generate(model, args, 1, "x")
    assert model.generate_calls[0]["guidance"] == 2.5
    assert args.guidance is None
