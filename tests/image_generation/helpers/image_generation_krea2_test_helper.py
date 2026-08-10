import os
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.utils.image_compare import ImageCompare

FOX_PROMPT = "a photograph of a red fox sitting in a sunlit forest clearing, sharp focus, bokeh"


class ImageGeneratorKrea2TestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        prompt: str = FOX_PROMPT,
        steps: int = 8,
        seed: int = 42,
        height: int = 1024,
        width: int = 1024,
        quantize: int | None = 8,
        guidance: float = 1.0,
        scheduler: str | None = None,
        mismatch_threshold: float | None = None,
    ) -> None:
        reference_image_path = ImageGeneratorKrea2TestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorKrea2TestHelper.resolve_path(output_image_path)

        try:
            model = Krea2(
                model_config=ModelConfig.krea2(),
                quantize=quantize,
            )
            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                scheduler=scheduler,
            )
            image.save(output_image_path)

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
