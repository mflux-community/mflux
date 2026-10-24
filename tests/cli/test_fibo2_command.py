import json
import sys

import PIL.Image
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo2.cli import (
    fibo2_edit_generate as edit_cli,
    fibo2_generate as cli,
)
from mflux.models.fibo2.variants import Fibo2, Fibo2Edit
from mflux.utils.exceptions import ModelConfigError
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

pytestmark = pytest.mark.fast


class FakeFibo2(FakeModel):
    real = Fibo2


class FakeFibo2Edit(FakeModel):
    real = Fibo2Edit


class FakeFiboVLM:
    # Records each caption it writes and how many FIBO 2 models existed then
    calls: list[tuple[str, int | None, int]] = []

    def __init__(self, quantize: int | None = None):
        self.quantize = quantize

    def generate(self, prompt: str, seed: int | None = None) -> str:
        FakeFiboVLM.calls.append((prompt, self.quantize, len(FakeFibo2.instances)))
        return json.dumps({"short_description": prompt, "aesthetics": {"aesthetic_score": "high"}})


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeFibo2.instances.clear()
    FakeFibo2Edit.instances.clear()
    FakeFiboVLM.calls.clear()
    monkeypatch.setattr(cli, "Fibo2", FakeFibo2)
    monkeypatch.setattr(cli, "FiboVLM", FakeFiboVLM)
    monkeypatch.setattr(edit_cli, "Fibo2Edit", FakeFibo2Edit)
    reset_cli_globals(monkeypatch)
    yield


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "source.png"
    PIL.Image.new("RGB", (64, 32)).save(path)
    return path


@pytest.fixture
def lora_file(tmp_path):
    path = tmp_path / "style.safetensors"
    path.write_bytes(b"")
    return path


@pytest.fixture
def shared_argv(tmp_path, lora_file):
    # Every value differs from its default, so a flag the command drops or hardcodes shows up
    return [
        "--prompt", "an owl",
        "--seed", "7", "8",
        "--steps", "12",
        "--guidance", "3.5",
        "--negative-prompt", "blurry",
        "--no-text-images",
        "--step-cache-ratio", "0.25",
        "--lora", str(lora_file), "0.5",
        "--no-bake-lora",
        "-q", "8",
        "--output", str(tmp_path / "out.png"),
    ]  # fmt: skip


def run(monkeypatch, module, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", *argv])
    module.main()


def exit_code(monkeypatch, capsys, module, argv: list[str]) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", *argv])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    return exit_info.value.code, capsys.readouterr().err


def expected_init(model_config: ModelConfig, lora_file) -> dict:
    return {
        "model_config": model_config,
        "quantize": 8,
        "model_path": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": False,
    }


def test_text_to_image_passes_every_flag_to_the_model(monkeypatch, tmp_path, lora_file, shared_argv):
    run(monkeypatch, cli, [*shared_argv, "--width", "512", "--height", "768"])
    model = FakeFibo2.instances[0]
    assert model.init_kwargs == expected_init(ModelConfig.fibo2(), lora_file)
    assert model.generate_calls == [
        {
            "seed": seed,
            "prompt": "an owl",
            "negative_prompt": "blurry",
            "width": 512,
            "height": 768,
            "guidance": 3.5,
            "num_inference_steps": 12,
            "render_text_images": False,
            "step_cache_ratio": 0.25,
        }
        for seed in (7, 8)
    ]
    assert [image.saves for image in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), False)],
        [(str(tmp_path / "out_seed_8.png"), False)],
    ]


def test_edit_passes_every_flag_to_the_model(monkeypatch, tmp_path, image, lora_file, shared_argv):
    mask = tmp_path / "mask.png"
    PIL.Image.new("L", (64, 32), 255).save(mask)
    argv = [*shared_argv, "--image-paths", str(image), "--mask-path", str(mask), "--width", "2x", "--height", "3x"]
    run(monkeypatch, edit_cli, argv)
    model = FakeFibo2Edit.instances[0]
    assert model.init_kwargs == expected_init(ModelConfig.fibo2(), lora_file)
    calls = [{**call, "image_paths": [str(p) for p in call["image_paths"]], "mask_path": str(call["mask_path"])} for call in model.generate_calls]  # fmt: off
    assert calls == [
        {
            "seed": seed,
            "prompt": "an owl",
            "image_paths": [str(image)],
            "mask_path": str(mask),
            "negative_prompt": "blurry",
            # 2x and 3x of the 64x32 source
            "width": 128,
            "height": 96,
            "guidance": 3.5,
            "num_inference_steps": 12,
            "render_text_images": False,
            "step_cache_ratio": 0.25,
        }
        for seed in (7, 8)
    ]
    assert [image.saves for image in model.images] == [
        [(str(tmp_path / "out_seed_7.png"), False)],
        [(str(tmp_path / "out_seed_8.png"), False)],
    ]


def test_text_to_image_defaults_leave_steps_at_50_and_guidance_to_the_model(monkeypatch, tmp_path):
    run(monkeypatch, cli, ["--prompt", "an owl", "--output", str(tmp_path / "out.png")])
    call = FakeFibo2.instances[0].generate_calls[0]
    assert (call["num_inference_steps"], call["guidance"], call["render_text_images"]) == (50, None, True)
    assert (call["width"], call["height"]) == (1024, 1024)


def test_edit_auto_size_is_left_to_the_model(monkeypatch, tmp_path, image):
    run(monkeypatch, edit_cli, ["--prompt", "p", "--image-paths", str(image), "--output", str(tmp_path / "o.png")])
    call = FakeFibo2Edit.instances[0].generate_calls[0]
    assert (call["width"], call["height"]) == (None, None)
    assert call["num_inference_steps"] == 50


def test_edit_turbo_defaults_to_4_steps(monkeypatch, tmp_path, image):
    argv = ["--prompt", "p", "--image-paths", str(image), "--model", "fibo2-edit-turbo"]
    run(monkeypatch, edit_cli, [*argv, "--output", str(tmp_path / "out.png")])
    model = FakeFibo2Edit.instances[0]
    assert model.model_config is ModelConfig.fibo2_edit_turbo()
    assert model.generate_calls[0]["num_inference_steps"] == 4


def test_text_to_image_runs_edit_turbo_with_its_defaults(monkeypatch, tmp_path):
    run(monkeypatch, cli, ["--prompt", "p", "--model", "fibo2-edit-turbo", "--output", str(tmp_path / "out.png")])
    model = FakeFibo2.instances[0]
    assert model.model_config is ModelConfig.fibo2_edit_turbo()
    # 4 steps, and guidance left to the model, which runs Edit Turbo at 1.0
    call = model.generate_calls[0]
    assert (call["num_inference_steps"], call["guidance"]) == (4, None)


def test_expand_prompt_writes_the_caption_once_before_fibo2_loads(monkeypatch, tmp_path):
    argv = ["--prompt", "an owl", "--expand-prompt", "--seed", "1", "2", "-q", "8", "--output", str(tmp_path / "o.png")]
    run(monkeypatch, cli, argv)
    assert FakeFiboVLM.calls == [("an owl", 8, 0)]
    caption = json.dumps({"short_description": "an owl", "aesthetics": {"aesthetic_score": "high"}})
    assert [call["prompt"] for call in FakeFibo2.instances[0].generate_calls] == [caption, caption]


def test_expand_prompt_keeps_a_json_caption_as_it_is(monkeypatch, tmp_path):
    prompt = '{"short_description": "an owl"}'
    run(monkeypatch, cli, ["--prompt", prompt, "--expand-prompt", "--output", str(tmp_path / "o.png")])
    assert FakeFiboVLM.calls == []
    assert FakeFibo2.instances[0].generate_calls[0]["prompt"] == prompt


def test_text_to_image_refuses_another_model_before_loading(monkeypatch, capsys):
    code, err = exit_code(monkeypatch, capsys, cli, ["--prompt", "p", "--model", "fibo"])
    assert code == 2
    assert "only accepts the aliases" in err
    assert FakeFibo2.instances == []
    with pytest.raises(ModelConfigError):
        cli.Fibo2Command.validate(cli.build_parser().parse_args())


@pytest.mark.parametrize("module", [cli, edit_cli], ids=["text-to-image", "edit"])
@pytest.mark.parametrize(("option", "value"), [("--scheduler", "linear"), ("--lora-style", "storyboard")])
def test_both_commands_warn_about_options_they_ignore(monkeypatch, tmp_path, image, module, option, value):
    assert option in module.IGNORED_OPTIONS
    argv = ["--prompt", "p", option, value, "--output", str(tmp_path / "o.png")]
    if module is edit_cli:
        argv += ["--image-paths", str(image)]
    with pytest.warns(UserWarning, match=f"{option} is ignored"):
        run(monkeypatch, module, argv)


@pytest.mark.parametrize(
    ("module", "extra"),
    [(cli, ["--guidance", "1"]), (edit_cli, ["--model", "fibo2-edit-turbo"])],
    ids=["dev-at-guidance-1", "turbo-default"],
)
def test_a_negative_prompt_at_guidance_1_is_reported(monkeypatch, tmp_path, image, module, extra):
    argv = ["--prompt", "p", "--negative-prompt", "blurry", *extra, "--output", str(tmp_path / "o.png")]
    if module is edit_cli:
        argv += ["--image-paths", str(image)]
    with pytest.warns(UserWarning, match="--negative-prompt is ignored at guidance 1"):
        run(monkeypatch, module, argv)


@pytest.mark.parametrize(
    ("module", "fake"),
    [(cli, FakeFibo2), (edit_cli, FakeFibo2Edit)],
    ids=["text-to-image", "edit"],
)
def test_edit_turbo_warns_when_guidance_turns_it_on(monkeypatch, tmp_path, image, module, fake):
    argv = ["--prompt", "p", "--model", "fibo2-edit-turbo", "--guidance", "4", "--output", str(tmp_path / "out.png")]
    if module is edit_cli:
        argv += ["--image-paths", str(image)]
    with pytest.warns(UserWarning, match="distilled to run without guidance"):
        run(monkeypatch, module, argv)
    assert fake.instances[0].generate_calls[0]["guidance"] == 4.0


@pytest.mark.parametrize("module", [cli, edit_cli], ids=["text-to-image", "edit"])
def test_a_sidecar_replays_the_step_cache(monkeypatch, tmp_path, image, module):
    sidecar = tmp_path / "image.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "p", "seed": 1, "steps": 50, "step_cache_ratio": 0.25}))
    argv = ["--config-from-metadata", str(sidecar)]
    if module is edit_cli:
        argv += ["--image-paths", str(image)]
    monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", *argv])
    assert module.build_parser().parse_args().step_cache_ratio == 0.25


@pytest.mark.parametrize("module", [cli, edit_cli], ids=["text-to-image", "edit"])
@pytest.mark.parametrize(
    ("recorded", "flag", "expected"),
    [(False, [], False), (False, ["--text-images"], True), (None, [], True), (None, ["--no-text-images"], False)],
)
def test_a_sidecar_replays_text_images_that_were_off(monkeypatch, tmp_path, image, module, recorded, flag, expected):
    sidecar = tmp_path / "image.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "p", "seed": 1, "steps": 4} | ({} if recorded is None else {"render_text_images": recorded})))  # fmt: off
    argv = ["--config-from-metadata", str(sidecar), *flag]
    if module is edit_cli:
        argv += ["--image-paths", str(image)]
    monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", *argv])
    assert module.build_parser().parse_args().render_text_images is expected


def test_a_sidecar_text_images_value_must_be_a_boolean(monkeypatch, capsys, tmp_path):
    sidecar = tmp_path / "image.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "p", "seed": 1, "render_text_images": "false"}))
    code, err = exit_code(monkeypatch, capsys, cli, ["--config-from-metadata", str(sidecar)])
    assert code == 2
    assert "render_text_images in --config-from-metadata must be true or false" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--prompt", '{"short_description": "x",}'],
            "not valid JSON: Illegal trailing comma before end of object at line 1, column 26",
        ),
        (["--prompt", '{"short_description": "x", "lighting": "soft"}'], "lighting has to be a JSON object"),
        (["--model", "fibo2-edit-turbo", "--prompt-file", "FILE"], "lighting has to be a JSON object"),
    ],
)
def test_text_to_image_refuses_a_broken_caption_before_loading(monkeypatch, capsys, tmp_path, argv, message):
    prompt_file = tmp_path / "caption.json"
    prompt_file.write_text('{"short_description": "x",\n "lighting": "soft"}')
    code, err = exit_code(monkeypatch, capsys, cli, [str(prompt_file) if value == "FILE" else value for value in argv])
    assert code == 2
    assert message in err
    assert FakeFibo2.instances == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--prompt", "p", "--image-paths", *["IMG"] * 6], "1 to 5 images"),
        (["--prompt", "p", "--image-paths", "IMG", "IMG", "--mask-path", "IMG"], "single image"),
        (["--image-paths", "IMG", "--prompt", '{"short_description": "x"}'], "edit_instruction"),
        (["--image-paths", "IMG", "--prompt-file", "JSON"], "edit_instruction"),
        (
            ["--image-paths", "IMG", "--prompt", '{"edit_instruction": "y", "objects": {"description": "d"}}'],
            "objects has to be a list",
        ),
    ],
)
def test_edit_refuses_bad_flags_and_prompts_before_loading(monkeypatch, capsys, tmp_path, image, extra, message):
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text('{"short_description": "x"}')
    argv = [str(image) if value == "IMG" else str(prompt_file) if value == "JSON" else value for value in extra]
    code, err = exit_code(monkeypatch, capsys, edit_cli, argv)
    assert code == 2
    assert message in err
    assert FakeFibo2Edit.instances == []
