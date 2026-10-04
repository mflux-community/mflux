import pytest

from tests.image_generation.helpers.image_generation_z_image_test_helper import ImageGeneratorZImageTestHelper

ASTRONAUT_CAT_PROMPT = (
    "A fluffy orange tabby cat wearing tiny astronaut helmet, "
    "floating in zero gravity inside a space station. "
    "Earth visible through the window behind. "
    "Photorealistic, cinematic lighting, 8k detail."
)

FROG_LORA_PROMPT = (
    "t3chnic4lly vibrant 1960s close-up of a woman sitting under a tree in a blue skirt and white blouse, "
    "she has blonde wavy short hair and a smile with green eyes lake scene by a garden with flowers in the foreground "
    "1960s style film She's holding her hand out there is a small smooth frog in her palm, "
    "she's making eye contact with the toad."
)


# The references without a suffix come from the float32 hidden stream that mflux used before #761.
# The --float32 option keeps that stream, so those references pin it. The _bf16 references pin the default.
STREAMS = [
    pytest.param(True, "", id="float32"),
    pytest.param(False, "_bf16", id="bf16"),
]


class TestImageGeneratorZImage:
    @pytest.mark.slow
    @pytest.mark.parametrize(("float32", "suffix"), STREAMS)
    def test_image_generation_z_image_turbo(self, float32, suffix):
        ImageGeneratorZImageTestHelper.assert_matches_reference_image(
            reference_image_path=f"reference_z_image_turbo{suffix}.png",
            output_image_path=f"output_z_image_turbo{suffix}.png",
            prompt=ASTRONAUT_CAT_PROMPT,
            steps=9,
            seed=42,
            height=368,
            width=640,
            quantize=8,
            float32=float32,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(("float32", "suffix"), STREAMS)
    def test_image_generation_z_image_turbo_lora(self, float32, suffix):
        ImageGeneratorZImageTestHelper.assert_matches_reference_image(
            reference_image_path=f"reference_z_image_turbo_lora{suffix}.png",
            output_image_path=f"output_z_image_turbo_lora{suffix}.png",
            prompt=FROG_LORA_PROMPT,
            steps=9,
            seed=42,
            height=368,
            width=640,
            quantize=8,
            lora_paths=["renderartist/Technically-Color-Z-Image-Turbo"],
            lora_scales=[0.5],
            clear_lora_cache_pattern="Technically-Color",  # Test fresh LoRA download works
            mismatch_threshold=0.35,  # LoRA tests have higher variance
            float32=float32,
        )
