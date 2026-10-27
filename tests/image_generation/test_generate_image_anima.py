import pytest

from mflux.models.common.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_anima_test_helper import ImageGeneratorAnimaTestHelper

# The model card's prompting: quality and safety tags first; no score tags for Aesthetic.
LANTERN_PROMPT = (
    "masterpiece, best quality, score_7, safe, 1girl, solo, long hair, silver hair, red eyes, witch hat, "
    "black cloak, holding lantern, night, forest, fireflies, looking at viewer, smile, upper body"
)
SIGN_PROMPT = (
    "masterpiece, best quality, safe, An anime girl wearing a black tank-top and denim shorts is standing outdoors. "
    "She's holding a rectangular sign out in front of her that reads \"ANIMA\". She's looking at the viewer with "
    "a smile. The background features some trees and blue sky with clouds."
)
NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"  # fmt: off
AESTHETIC_NEGATIVE_PROMPT = "worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration"


class TestImageGeneratorAnima:
    @pytest.mark.slow
    def test_image_generation_anima_turbo(self):
        ImageGeneratorAnimaTestHelper.assert_matches_reference_image(
            reference_image_path="reference_anima_turbo.png",
            output_image_path="output_anima_turbo.png",
            model_config=ModelConfig.anima_turbo(),
            prompt=LANTERN_PROMPT,
            steps=10,
            seed=42,
            height=1024,
            width=1024,
        )

    @pytest.mark.slow
    def test_image_generation_anima_aesthetic(self):
        ImageGeneratorAnimaTestHelper.assert_matches_reference_image(
            reference_image_path="reference_anima_aesthetic.png",
            output_image_path="output_anima_aesthetic.png",
            model_config=ModelConfig.anima_aesthetic(),
            prompt=SIGN_PROMPT,
            negative_prompt=AESTHETIC_NEGATIVE_PROMPT,
            steps=30,
            seed=42,
            height=1024,
            width=768,
        )

    @pytest.mark.slow
    def test_image_generation_anima_base_euler(self):
        # Base is stored with the net. key prefix, unlike the other two files; euler is the other sampler.
        ImageGeneratorAnimaTestHelper.assert_matches_reference_image(
            reference_image_path="reference_anima_base_euler.png",
            output_image_path="output_anima_base_euler.png",
            model_config=ModelConfig.anima_base(),
            prompt=LANTERN_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            steps=30,
            seed=42,
            height=768,
            width=768,
            scheduler="euler",
        )
