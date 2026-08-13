import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2KleinEdit
from tests.image_generation.helpers.image_generation_test_helper import ImageGeneratorTestHelper


class TestImageGeneratorFlux2KleinEdit:
    @pytest.mark.slow
    def test_flux2_klein_9b_edit_dalmatian(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux2_klein_edit_dalmatian.png",
            output_image_path="output_flux2_klein_edit_dalmatian.png",
            model_class=Flux2KleinEdit,
            model_config=ModelConfig.flux2_klein_9b(),
            quantize=None,
            steps=4,
            seed=42,
            height=1024,
            width=640,
            guidance=1.0,
            image_paths=["unsplash_dog.jpg"],
            prompt="Make the dog a dalmatian",
            mismatch_threshold=0.15,
        )

    @pytest.mark.slow
    def test_flux2_klein_9b_kv_edit_dalmatian(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux2_klein_9b_kv_edit_dalmatian.png",
            output_image_path="output_flux2_klein_9b_kv_edit_dalmatian.png",
            model_class=Flux2KleinEdit,
            model_config=ModelConfig.flux2_klein_9b_kv(),
            quantize=None,
            steps=4,
            seed=42,
            height=1024,
            width=640,
            guidance=1.0,
            image_paths=["unsplash_dog.jpg"],
            prompt="Make the dog a dalmatian",
            mismatch_threshold=0.15,
        )

    @pytest.mark.slow
    def test_flux2_klein_9b_edit_sunglasses_wide(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux2_klein_edit_sunglasses_wide.png",
            output_image_path="output_flux2_klein_edit_sunglasses_wide.png",
            model_class=Flux2KleinEdit,
            model_config=ModelConfig.flux2_klein_9b(),
            quantize=None,
            steps=4,
            seed=42,
            height=896,
            width=1344,
            guidance=1.0,
            image_paths=["unsplash_person.jpg"],
            prompt="put on sunglasses",
            mismatch_threshold=0.15,
        )

    @pytest.mark.slow
    def test_flux2_klein_9b_edit_glasses_wide(self):
        ImageGeneratorTestHelper.assert_matches_reference_image(
            reference_image_path="reference_flux2_klein_edit_glasses_wide.png",
            output_image_path="output_flux2_klein_edit_glasses_wide.png",
            model_class=Flux2KleinEdit,
            model_config=ModelConfig.flux2_klein_9b(),
            quantize=None,
            steps=4,
            seed=42,
            height=896,
            width=1344,
            guidance=1.0,
            image_paths=["unsplash_person.jpg", "glasses.jpg"],
            prompt="Make the woman wear the eyeglasses (regular glasses, not sunglasses)",
            mismatch_threshold=0.15,
        )
