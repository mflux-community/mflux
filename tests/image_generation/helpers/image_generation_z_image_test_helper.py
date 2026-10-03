import os
import shutil
from pathlib import Path

from mflux.cli.defaults.defaults import MFLUX_LORA_CACHE_DIR
from mflux.models.z_image import ZImageTurbo
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorZImageTestHelper:
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
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        mismatch_threshold: float | None = None,
        clear_lora_cache_pattern: str | None = None,
        float32: bool = False,
    ):
        reference_image_path = ImageGeneratorZImageTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorZImageTestHelper.resolve_path(output_image_path)

        # Clear cached LoRA to test download functionality
        if clear_lora_cache_pattern:
            ImageGeneratorZImageTestHelper.clear_cached_lora(clear_lora_cache_pattern)

        try:
            model = ZImageTurbo(
                quantize=quantize,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                float32=float32,
            )

            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
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
    def clear_cached_lora(pattern: str) -> None:
        cache_dir = MFLUX_LORA_CACHE_DIR
        if not cache_dir.exists():
            return

        # Look for directories matching the pattern (HuggingFace style: models--repo--name)
        for item in cache_dir.iterdir():
            if pattern.lower() in item.name.lower():
                if item.is_dir():
                    print(f"🗑️  Clearing cached LoRA directory: {item}")
                    shutil.rmtree(item)
                elif item.is_file() or item.is_symlink():
                    print(f"🗑️  Clearing cached LoRA file: {item}")
                    item.unlink()

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
