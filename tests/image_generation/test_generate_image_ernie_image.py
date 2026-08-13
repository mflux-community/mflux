import pytest

from mflux.models.common.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_ernie_image_test_helper import ImageGeneratorErnieImageTestHelper

BICYCLE_PROMPT = (
    "A vintage red bicycle leaning against a weathered cobblestone wall covered in green ivy, "
    "narrow European alley, warm golden afternoon sunlight, shallow depth of field, photorealistic."
)


class TestImageGeneratorErnieImage:
    @pytest.mark.slow
    def test_image_generation_ernie_image_turbo(self):
        ImageGeneratorErnieImageTestHelper.assert_matches_reference_image(
            reference_image_path="reference_ernie_image_turbo.png",
            output_image_path="output_ernie_image_turbo.png",
            prompt=BICYCLE_PROMPT,
            steps=8,
            seed=7,
            height=368,
            width=640,
            guidance=1.0,
            quantize=8,
        )

    @pytest.mark.slow
    def test_image_generation_ernie_image(self):
        ImageGeneratorErnieImageTestHelper.assert_matches_reference_image(
            reference_image_path="reference_ernie_image.png",
            output_image_path="output_ernie_image.png",
            prompt=BICYCLE_PROMPT,
            steps=28,
            seed=7,
            height=368,
            width=640,
            guidance=4.0,
            quantize=8,
            model_config=ModelConfig.ernie_image(),
        )
