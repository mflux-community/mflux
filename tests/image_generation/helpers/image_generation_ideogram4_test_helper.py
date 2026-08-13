import gc
import os
from pathlib import Path
from typing import Any

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.ideogram4.variants import Ideogram4
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorIdeogram4TestHelper:
    JAZZ_FEST_JSON_CAPTION: dict[str, Any] = {
        "high_level_description": "A bold typographic event poster for a New Orleans jazz festival featuring a trumpet player silhouette.",
        "style_description": {
            "aesthetics": "dramatic, high contrast, vintage",
            "lighting": "strong stage spotlight from above, deep surrounding shadows",
            "medium": "graphic_design",
            "art_style": "screenprint aesthetic, limited color palette, bold geometric shapes",
            "color_palette": ["#0A0A0A", "#F5C518", "#E63946", "#FFFFFF"],
        },
        "compositional_deconstruction": {
            "background": "Near-black background with subtle aged paper texture.",
            "elements": [
                {
                    "type": "obj",
                    "bbox": [200, 260, 800, 740],
                    "desc": "A silhouette of a trumpet player mid-performance, arm raised, dramatic pose, rendered in deep gold against the dark background, centered in the poster.",
                },
                {
                    "type": "text",
                    "bbox": [35, 60, 150, 940],
                    "text": "NEW ORLEANS JAZZ FEST",
                    "desc": "Bold uppercase serif headline in bright white spanning the top of the poster.",
                },
                {
                    "type": "text",
                    "bbox": [850, 220, 930, 780],
                    "text": "JULY 12 · ARMSTRONG PARK",
                    "desc": "Smaller red sans-serif text with the date and venue along the lower edge.",
                },
            ],
        },
    }

    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        *,
        seed: int,
        height: int,
        width: int,
        preset: str = "V4_DEFAULT_20",
        num_inference_steps: int | None = None,
        guidance: float | None = None,
        quantize: int | None = None,
        mismatch_threshold: float | None = None,
        model_config: ModelConfig | None = None,
        model_path: str | None = None,
        prompt: str | dict[str, Any] | None = None,
    ) -> None:
        reference_image_path = ImageGeneratorIdeogram4TestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorIdeogram4TestHelper.resolve_path(output_image_path)

        model = None
        try:
            model = Ideogram4(
                quantize=quantize,
                model_path=model_path,
                model_config=model_config or ModelConfig.ideogram4_fp8(),
            )
            result = model.generate_image(
                seed=seed,
                prompt=prompt or ImageGeneratorIdeogram4TestHelper.JAZZ_FEST_JSON_CAPTION,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance=guidance,
                preset=preset,
            )
            result.save(path=output_image_path, overwrite=True)

            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
                mismatch_threshold=mismatch_threshold,
            )
        finally:
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)

            model = None
            gc.collect()

            try:
                import mlx.core as mx  # noqa: PLC0415

                mx.metal.clear_cache()
            except (ImportError, AttributeError):
                pass

    @staticmethod
    def resolve_path(path: str | Path) -> Path:
        return Path(__file__).parent.parent.parent / "resources" / path
