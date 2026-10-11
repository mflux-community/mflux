import pytest

from mflux.models.common.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_qwen_image_21_test_helper import (
    ImageGeneratorQwenImage21TestHelper,
)

PROMPT = (
    "Close-up portrait of a majestic tiger in its natural habitat, detailed fur texture, "
    "piercing eyes, natural forest background, soft natural lighting, wildlife photography, "
    "photorealistic, high detail, professional wildlife shot"
)


class TestImageGeneratorQwenImage21:
    @pytest.mark.slow
    def test_image_generation_qwen_image_21(self):
        ImageGeneratorQwenImage21TestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_image_21.png",
            output_image_path="output_qwen_image_21.png",
            prompt=PROMPT,
            steps=40,
            seed=42,
            height=320,
            width=512,
            quantize=8,
        )

    @pytest.mark.slow
    def test_image_generation_qwen_image_21_turbo(self):
        ImageGeneratorQwenImage21TestHelper.assert_matches_reference_image(
            reference_image_path="reference_qwen_image_21_turbo.png",
            output_image_path="output_qwen_image_21_turbo.png",
            prompt=PROMPT,
            steps=8,
            seed=42,
            # Turbo breaks fine detail into a pixel grid well below its trained resolutions, so
            # this test runs at 1024x1024, not the base test's 512x320.
            height=1024,
            width=1024,
            quantize=8,
            model_config=ModelConfig.qwen_image_21_turbo(),
        )
