import json
import sys

import mlx.core as mx
import pytest

from mflux.models.boogu.cli import boogu_image_generate
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.utils.image_util import ImageUtil


def _generated_image():
    config = Config(model_config=ModelConfig.boogu_image_turbo(), num_inference_steps=4, height=64, width=64)
    return ImageUtil.to_image(
        decoded_latents=mx.zeros((1, 3, 64, 64)),
        config=config,
        seed=7,
        prompt="a red cube",
        quantization=8,
        generation_time=1.0,
    )


@pytest.mark.fast
def test_the_cli_writes_generation_parameters_to_the_sidecar(tmp_path, monkeypatch):
    # The CLI used to hand its GeneratedImage to ImageUtil.save_image as if it were a PIL
    # image. That call re-entered GeneratedImage.save through image.save(path), so the
    # embedded metadata survived while the outer call wrote the sidecar with metadata=None,
    # and json.dump turned it into the four bytes "null".
    generated = _generated_image()

    class _StubModel:
        def __init__(self, **kwargs):
            pass

        def generate_image(self, **kwargs):
            return generated

    monkeypatch.setattr(boogu_image_generate, "BooguImage", _StubModel)
    monkeypatch.setattr(boogu_image_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    output = tmp_path / "out.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mflux-generate-boogu",
            "--prompt",
            "a red cube",
            "--seed",
            "7",
            "--steps",
            "4",
            "--metadata",
            "--output",
            str(output),
        ],
    )

    boogu_image_generate.main()

    sidecar = json.loads(output.with_suffix(".metadata.json").read_text())
    assert sidecar["seed"] == 7
    assert sidecar["prompt"] == "a red cube"
