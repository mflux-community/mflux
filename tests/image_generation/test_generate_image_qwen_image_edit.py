import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit import QwenImageEdit
from tests.image_generation.helpers.image_generation_edit_test_helper import ImageGeneratorEditTestHelper

# Pinned to the checkpoint by name rather than resolved through the bare
# "qwen-image-edit" alias. The alias has already moved between 2509 and 2511 and
# back (#474, #574); while it pointed elsewhere these references described a
# checkpoint the tests were no longer running, and nothing reported it. Naming the
# repo means a registry edit can invalidate this test loudly instead of silently.
# A 2511 test belongs beside this one, with its own reference and its own name.
QWEN_IMAGE_EDIT_2509 = ModelConfig.from_name("Qwen/Qwen-Image-Edit-2509")


class TestImageGeneratorQwenImageEdit:
    @pytest.mark.slow
    def test_image_generation_qwen_edit(self):
        ImageGeneratorEditTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_edit.png",
            output_image_path="output_qwen_edit.png",
            model_class=QwenImageEdit,
            model_config=QWEN_IMAGE_EDIT_2509,
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            quantize=8,
            prompt="Make the hand fistbump the camera instead of showing a flat palm",
            image_path="reference_upscaled.png",
            mismatch_threshold=0.25,
            low_memory=True,
        )

    @pytest.mark.slow
    def test_image_generation_qwen_edit_multiple_images(self):
        ImageGeneratorEditTestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_edit_multiple_images.png",
            output_image_path="output_qwen_edit_multiple_images.png",
            model_class=QwenImageEdit,
            model_config=QWEN_IMAGE_EDIT_2509,
            steps=20,
            seed=4869845,
            height=384,
            width=640,
            guidance=2.5,
            quantize=8,
            prompt="Make the hand fistbump the camera instead of showing a flat palm, and the man should wear this shirt. Maintain the original pose, body position, and overall stance.",
            image_paths=["reference_upscaled.png", "shirt.jpg"],
        )
