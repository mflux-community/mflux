from pathlib import Path

from PIL import Image

from mflux.models.dreamlite.variants.dreamlite_core import _DreamLiteCore
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class DreamLiteEdit(_DreamLiteCore):
    # An edit runs on a 1024x1024 copy of the source image, as the reference does, whatever its aspect ratio
    EDIT_SIZE = 1024

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_path: Path | str,
        num_inference_steps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        guidance: float | None = None,
        image_guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        image = ImageUtil.load_image(image_path)
        generated = self._generate(
            seed=seed,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=DreamLiteEdit.EDIT_SIZE,
            width=DreamLiteEdit.EDIT_SIZE,
            guidance=guidance,
            image_guidance=image_guidance,
            negative_prompt=negative_prompt,
            image=image,
            image_path=image_path,
        )
        # The result goes back to the source image's size, or to the size asked for
        output_size = (width or image.width, height or image.height)
        if generated.image.size != output_size:
            generated.image = generated.image.resize(output_size, Image.Resampling.LANCZOS)
            generated.width, generated.height = output_size
        return generated
