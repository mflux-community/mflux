from mflux.models.fibo2.variants.fibo2_core import _Fibo2Core
from mflux.utils.generated_image import GeneratedImage


class Fibo2(_Fibo2Core):
    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        render_text_images: bool = True,
    ) -> GeneratedImage:
        return self._generate(
            seed=seed,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            negative_prompt=negative_prompt,
            render_text_images=render_text_images,
        )
