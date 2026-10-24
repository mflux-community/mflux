from pathlib import Path

from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption_check import Fibo2CaptionCheck
from mflux.models.fibo2.variants.edit.fibo2_edit_images import Fibo2EditImages
from mflux.models.fibo2.variants.fibo2_core import _Fibo2Core
from mflux.utils.generated_image import GeneratedImage


class Fibo2Edit(_Fibo2Core):
    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: list[Path | str],
        mask_path: Path | str | None = None,
        num_inference_steps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        render_text_images: bool = True,
        step_cache_ratio: float | None = None,
    ) -> GeneratedImage:
        Fibo2CaptionCheck.check_and_log(prompt, edit=True)
        # Without width and height the result takes the trained size for the first image's aspect ratio
        images, text_encoder_images, (width, height) = Fibo2EditImages.prepare(image_paths, width, height, mask_path)
        return self._generate(
            seed=seed,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            negative_prompt=negative_prompt,
            render_text_images=render_text_images,
            step_cache_ratio=step_cache_ratio,
            images=images,
            text_encoder_images=text_encoder_images,
            image_paths=image_paths,
            mask_path=mask_path,
        )
