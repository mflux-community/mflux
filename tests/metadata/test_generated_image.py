import json

import mlx.core as mx
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.utils.generated_image import GeneratedImage


def test_fibo_edit_save_also_writes_prompt_json(tmp_path):
    output_path = tmp_path / "fibo_edit_output.png"
    prompt = json.dumps(
        {
            "short_description": "A white cat portrait",
            "edit_instruction": "Turn the cat color to white",
        }
    )

    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.fibo_edit(),
        seed=42,
        prompt=prompt,
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=16,
        width=16,
    )

    generated_image.save(path=output_path, overwrite=True)

    prompt_path = output_path.with_suffix(".json")
    assert output_path.exists()
    assert prompt_path.exists()
    assert json.loads(prompt_path.read_text()) == json.loads(prompt)


def test_exported_metadata_uses_metadata_sidecar_suffix(tmp_path):
    output_path = tmp_path / "generated.png"
    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.qwen_image(),
        seed=42,
        prompt="test prompt",
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=16,
        width=16,
    )

    generated_image.save(path=output_path, overwrite=True, export_json_metadata=True)

    metadata_path = output_path.with_suffix(".metadata.json")
    assert metadata_path.exists()
    assert not output_path.with_suffix(".json").exists()
    assert json.loads(metadata_path.read_text())["seed"] == 42


@pytest.fixture(autouse=True)
def _reset_run_model_path():
    # GeneratedImage.model_path is per-run state set by CommandLineParser.parse_args;
    # arg-parser tests in the same session parse custom checkpoints and would leak
    # their path into the writers built directly here.
    GeneratedImage.model_path = None
    yield
    GeneratedImage.model_path = None


def test_builtin_run_stores_null_base_model(tmp_path):
    # str(None) used to land here as the string "None", which --config-from-metadata then
    # fed to the --base-model validator (#695). A builtin run stores null; a custom
    # checkpoint stores the base it resolved to, which the validator accepts.
    output_path = tmp_path / "generated.png"
    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.dev(),
        seed=42,
        prompt="test prompt",
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=16,
        width=16,
    )

    generated_image.save(path=output_path, overwrite=True, export_json_metadata=True)

    stored = json.loads(output_path.with_suffix(".metadata.json").read_text())
    assert stored["base_model"] is None
    assert stored["model"] == ModelConfig.dev().model_name
    assert stored["model_path"] is None


def test_pid_decode_and_resize_round_trip_through_metadata(tmp_path):
    output_path = tmp_path / "pid_output.png"
    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.qwen_image(),
        seed=42,
        prompt="test prompt",
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=512,
        width=512,
        pid_decode=True,
        pid_degrade_sigma=0.2,
    )

    generated_image.save(path=output_path, overwrite=True, export_json_metadata=True)

    metadata = json.loads(output_path.with_suffix(".metadata.json").read_text())
    # Generation dims are kept as-is (they're what reproduces the run); the PiD flags
    # are what --config-from-metadata needs to know a 4x-larger PNG was actually produced.
    assert metadata["height"] == 512
    assert metadata["width"] == 512
    assert metadata["pid_decode"] is True
    assert metadata["pid_degrade_sigma"] == 0.2


def test_pid_flags_are_absent_from_metadata_without_pid_decode(tmp_path):
    output_path = tmp_path / "no_pid_output.png"
    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.qwen_image(),
        seed=42,
        prompt="test prompt",
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=512,
        width=512,
    )

    generated_image.save(path=output_path, overwrite=True, export_json_metadata=True)

    metadata = json.loads(output_path.with_suffix(".metadata.json").read_text())
    assert "pid_decode" not in metadata
    assert "pid_degrade_sigma" not in metadata


def test_get_right_half_keeps_every_metadata_attribute():
    original = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.dev(),
        seed=42,
        prompt="test prompt",
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        lora_paths=["style.safetensors"],
        lora_scales=[0.8],
        height=16,
        width=16,
        controlnet_image_path="control.png",
        controlnet_strength=0.6,
        image_path="init.png",
        image_paths=["ref1.png", "ref2.png"],
        image_strength=0.4,
        masked_image_path="mask.png",
        depth_image_path="depth.png",
        redux_image_paths=["redux1.png", "redux2.png"],
        redux_image_strengths=[0.5, 0.7],
        concept_heatmap=None,
        negative_prompt="blurry",
        init_metadata={"source": "test"},
        pid_decode=True,
        pid_degrade_sigma=0.2,
    )

    half = original.get_right_half()

    # The right half is the deliverable of the in-context commands, so its sidecar must
    # reproduce the run. Compare every stored attribute except the cropped image itself.
    original_attrs = {k: v for k, v in vars(original).items() if k != "image"}
    half_attrs = {k: v for k, v in vars(half).items() if k != "image"}
    assert half_attrs == original_attrs
    assert half.image.size == (8, 16)


def test_redux_image_paths_export_as_a_list(tmp_path):
    output_path = tmp_path / "redux_output.png"
    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.dev(),
        seed=42,
        prompt="test prompt",
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=16,
        width=16,
        redux_image_paths=["redux1.png", "redux2.png"],
        redux_image_strengths=[0.5, 0.7],
    )

    generated_image.save(path=output_path, overwrite=True, export_json_metadata=True)

    metadata = json.loads(output_path.with_suffix(".metadata.json").read_text())
    # Was str(list), i.e. a Python repr the restore side could never parse back.
    assert metadata["redux_image_paths"] == ["redux1.png", "redux2.png"]
    assert metadata["image_paths"] is None


def test_fibo_edit_save_keeps_prompt_json_and_exports_metadata_separately(tmp_path):
    output_path = tmp_path / "fibo_edit_output.png"
    prompt = json.dumps(
        {
            "short_description": "A white cat portrait",
            "edit_instruction": "Turn the cat color to white",
        }
    )

    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.fibo_edit(),
        seed=42,
        prompt=prompt,
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=16,
        width=16,
    )

    generated_image.save(path=output_path, overwrite=True, export_json_metadata=True)

    prompt_path = output_path.with_suffix(".json")
    metadata_path = output_path.with_suffix(".metadata.json")
    assert prompt_path.exists()
    assert metadata_path.exists()
    assert json.loads(prompt_path.read_text()) == json.loads(prompt)
    assert json.loads(metadata_path.read_text())["prompt"] == prompt


def test_fibo_edit_prompt_json_tracks_final_output_name_when_image_exists(tmp_path):
    output_path = tmp_path / "fibo_edit_output.png"
    output_path.write_bytes(b"existing image")
    prompt = json.dumps(
        {
            "short_description": "A white cat portrait",
            "edit_instruction": "Turn the cat color to white",
        }
    )

    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.fibo_edit(),
        seed=42,
        prompt=prompt,
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=16,
        width=16,
    )

    generated_image.save(path=output_path, overwrite=False)

    final_output_path = tmp_path / "fibo_edit_output_1.png"
    final_prompt_path = tmp_path / "fibo_edit_output_1.json"
    assert final_output_path.exists()
    assert final_prompt_path.exists()
    assert not (tmp_path / "fibo_edit_output.json").exists()
    assert json.loads(final_prompt_path.read_text()) == json.loads(prompt)


def test_custom_checkpoint_stores_the_base_it_resolved_to(tmp_path):
    # The other half of the #695 fix, previously unpinned: a custom checkpoint declared
    # with --base-model stores that base, and the stored value is one the --base-model
    # validator accepts on replay.
    from mflux.models.common.resolution.config_resolution import ConfigResolution

    output_path = tmp_path / "generated.png"
    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.from_name("some-org/dev-finetune", base_model="dev"),
        seed=42,
        prompt="test prompt",
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=16,
        width=16,
    )

    generated_image.save(path=output_path, overwrite=True, export_json_metadata=True)

    stored = json.loads(output_path.with_suffix(".metadata.json").read_text())
    assert stored["base_model"] == "black-forest-labs/FLUX.1-dev"
    assert stored["base_model"] in ConfigResolution.base_model_names()


def test_the_sidecar_records_the_weights_source_of_the_run(tmp_path):
    # A local-checkpoint run resolves to a registry entry, and "model" stores that entry:
    # without the source path next to it the run reads back as the builtin and -C replays
    # it against the registry weights (#705). The parser sets this per run; unset (a
    # library caller, or a builtin run) stores null.
    output_path = tmp_path / "generated.png"
    GeneratedImage.model_path = "/models/klein-4b-q4"
    generated_image = GeneratedImage(
        image=Image.new("RGB", (16, 16), "white"),
        model_config=ModelConfig.dev(),
        seed=42,
        prompt="test prompt",
        steps=20,
        guidance=3.5,
        precision=mx.bfloat16,
        quantization=8,
        generation_time=1.23,
        height=16,
        width=16,
    )

    generated_image.save(path=output_path, overwrite=True, export_json_metadata=True)

    stored = json.loads(output_path.with_suffix(".metadata.json").read_text())
    assert stored["model_path"] == "/models/klein-4b-q4"
