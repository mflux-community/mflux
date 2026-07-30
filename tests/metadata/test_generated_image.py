import json

import mlx.core as mx
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


def test_pid_flags_are_null_in_metadata_without_pid_decode(tmp_path):
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
    assert metadata["pid_decode"] is False
    assert metadata["pid_degrade_sigma"] is None


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
