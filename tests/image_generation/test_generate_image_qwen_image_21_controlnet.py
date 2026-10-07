import os
from pathlib import Path

import mlx.core as mx
import pytest

from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet
from mflux.utils.image_compare import ImageCompare

RESOURCES = Path(__file__).parent.parent / "resources"
# Canny edges of reference_upscaled.png: a man holding his open hand toward the camera.
CONTROL = RESOURCES / "qwen21_controlnet_canny.png"
PROMPT = "A man in a white t-shirt holds his open hand toward the camera, soft window light, photograph"


@pytest.fixture(scope="module")
def model():
    # One q8 load shared by the module: the base model plus the 16 control blocks, about 28 GB at its peak.
    model = QwenImage21Controlnet(quantize=8)
    yield model
    del model
    mx.clear_cache()


@pytest.mark.slow
class TestImageGeneratorQwenImage21Controlnet:
    def test_canny_control(self, model, tmp_path):
        reference = "reference_qwen_image_21_controlnet_canny.png"
        output = tmp_path / f"output_{reference}"
        if "MFLUX_PRESERVE_TEST_OUTPUT" in os.environ:
            output = RESOURCES / f"output_{reference}"
        image = model.generate_image(
            seed=43, prompt=PROMPT, controlnet_image_path=str(CONTROL), num_inference_steps=20, width=640, height=384
        )
        image.save(path=output, overwrite=True)
        ImageCompare.check_images_close_enough(
            output,
            RESOURCES / reference,
            "Generated Qwen-Image-2.1 ControlNet image doesn't match reference image.",
            mismatch_threshold=0.25,
        )
