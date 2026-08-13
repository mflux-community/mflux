import pytest

from mflux.models.common.config import ModelConfig
from tests.image_generation.helpers.image_generation_krea2_test_helper import (
    FOX_PROMPT,
    ImageGeneratorKrea2TestHelper,
)


class TestImageGeneratorKrea2:
    @pytest.mark.slow
    def test_image_generation_krea2_turbo(self):
        ImageGeneratorKrea2TestHelper.assert_matches_reference_image(
            reference_image_path="reference_krea2_turbo.png",
            output_image_path="output_krea2_turbo.png",
            prompt=FOX_PROMPT,
            steps=8,
            seed=42,
            height=1024,
            width=1024,
            quantize=8,
            guidance=1.0,
            scheduler="er_sde",
        )

    @pytest.mark.slow
    def test_image_generation_krea2_raw(self):
        # Krea 2 Raw is the un-distilled base (train LoRAs on it, per Krea); it still generates
        # coherently at guidance 1.0 with more steps than Turbo. Exercises the krea-2-raw config
        # and the Raw loading path end to end.
        ImageGeneratorKrea2TestHelper.assert_matches_reference_image(
            reference_image_path="reference_krea2_raw.png",
            output_image_path="output_krea2_raw.png",
            prompt=FOX_PROMPT,
            steps=28,
            seed=42,
            height=1024,
            width=1024,
            quantize=8,
            guidance=1.0,
            scheduler="er_sde",
            model_config=ModelConfig.krea2_raw(),
        )
