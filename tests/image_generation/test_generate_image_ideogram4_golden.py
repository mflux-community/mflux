import pytest

from tests.image_generation.helpers.image_generation_ideogram4_test_helper import (
    ImageGeneratorIdeogram4TestHelper,
)


class TestImageGeneratorIdeogram4Golden:
    @pytest.mark.slow
    def test_ideogram4_jazz_fest_json_caption(self):
        ImageGeneratorIdeogram4TestHelper.assert_matches_reference_image(
            reference_image_path="reference_ideogram4_jazz_fest.png",
            output_image_path="output_ideogram4_jazz_fest.png",
            prompt=ImageGeneratorIdeogram4TestHelper.JAZZ_FEST_JSON_CAPTION,
            seed=202,
            width=768,
            height=576,
            preset="V4_TURBO_12",
            quantize=None,
            mismatch_threshold=0.15,
        )
