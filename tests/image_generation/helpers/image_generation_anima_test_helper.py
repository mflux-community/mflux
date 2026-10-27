import os
from pathlib import Path

from mflux.models.anima.variants.txt2img.anima import Anima
from mflux.models.common.config.model_config import ModelConfig
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorAnimaTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_config: ModelConfig,
        prompt: str,
        steps: int,
        seed: int,
        height: int,
        width: int,
        negative_prompt: str | None = None,
        scheduler: str | None = None,
        quantize: int | None = 8,
        mismatch_threshold: float | None = None,
    ):
        reference_image_path = ImageGeneratorAnimaTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorAnimaTestHelper.resolve_path(output_image_path)

        try:
            model = Anima(model_config=model_config, quantize=quantize)

            result = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                scheduler=scheduler,
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
