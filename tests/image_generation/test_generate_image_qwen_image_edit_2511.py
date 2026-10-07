import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit import QwenImageEdit
from tests.image_generation.helpers.image_generation_edit_test_helper import ImageGeneratorEditTestHelper

# The same requests as the 2509 references beside this file, on the 2511 checkpoint, pinned
# by name for the reason test_generate_image_qwen_image_edit.py gives.
QWEN_IMAGE_EDIT_2511 = ModelConfig.from_name("Qwen/Qwen-Image-Edit-2511")


class TestImageGeneratorQwenImageEdit2511:
    @pytest.mark.slow
    def test_image_generation_qwen_edit_2511(self):
        ImageGeneratorEditTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_edit_2511.png",
            output_image_path="output_qwen_edit_2511.png",
            model_class=QwenImageEdit,
            model_config=QWEN_IMAGE_EDIT_2511,
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            quantize=8,
            prompt="Make the hand fistbump the camera instead of showing a flat palm",
            image_path="reference_upscaled.png",
            low_memory=True,
        )

    @pytest.mark.slow
    def test_image_generation_qwen_edit_2511_multiple_images(self):
        ImageGeneratorEditTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_edit_2511_multiple_images.png",
            output_image_path="output_qwen_edit_2511_multiple_images.png",
            model_class=QwenImageEdit,
            model_config=QWEN_IMAGE_EDIT_2511,
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            quantize=8,
            prompt="Make the hand fistbump the camera instead of showing a flat palm, and the man should wear this shirt. Maintain the original pose, body position, and overall stance.",
            image_paths=["reference_upscaled.png", "shirt.jpg"],
        )
