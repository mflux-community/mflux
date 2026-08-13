import pytest

from tests.image_generation.helpers.image_generation_lens_test_helper import ImageGeneratorLensTestHelper


@pytest.mark.slow
class TestImageGeneratorLens:
    def test_lens_turbo_bookshop_cat(self):
        ImageGeneratorLensTestHelper.assert_matches_reference_image(
            reference_image_path="reference_lens_turbo.png",
            output_image_path="output_lens_turbo.png",
            prompt="A cozy bookshop cafe at dusk, warm light through the window, a cat sleeping on a stack of books",
            steps=4,
            seed=7,
            height=512,
            width=512,
        )
