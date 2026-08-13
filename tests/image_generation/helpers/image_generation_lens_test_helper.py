import os
from pathlib import Path

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.lens.variants.txt2img.lens_image import LensImage
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorLensTestHelper:
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
        model_config: ModelConfig | None = None,
    ):
        reference_image_path = ImageGeneratorLensTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorLensTestHelper.resolve_path(output_image_path)

        try:
            model = LensImage(quantize=quantize, model_config=model_config or ModelConfig.lens_turbo())

            result = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
            )

            result.image.save(output_image_path)

            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
                mismatch_threshold=mismatch_threshold,
            )
        finally:
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
