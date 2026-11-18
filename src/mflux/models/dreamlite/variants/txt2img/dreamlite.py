from mflux.models.dreamlite.variants.dreamlite_core import _DreamLiteCore
from mflux.utils.generated_image import GeneratedImage


class DreamLite(_DreamLiteCore):
    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        return self._generate(
            seed=seed,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            negative_prompt=negative_prompt,
        )
