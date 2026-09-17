from pathlib import Path

from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorQwenImage21TestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        prompt: str,
        steps: int,
        seed: int,
        height: int,
        width: int,
        quantize: int | None = None,
        mismatch_threshold: float | None = None,
    ):
        reference_image_path = ImageGeneratorQwenImage21TestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorQwenImage21TestHelper.resolve_path(output_image_path)

        try:
            model = QwenImage21(
                quantize=quantize,
            )

            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
            )

            image.save(output_image_path, overwrite=True)

            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
                mismatch_threshold=mismatch_threshold,
            )
        finally:
            import os

            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
