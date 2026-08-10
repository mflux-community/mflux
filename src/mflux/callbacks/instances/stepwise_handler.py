from pathlib import Path

import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.callback import BeforeLoopCallback, InLoopCallback, InterruptCallback
from mflux.models.common.config.config import Config
from mflux.utils.image_util import ImageUtil


class StepwiseHandler(BeforeLoopCallback, InLoopCallback, InterruptCallback):
    def __init__(
        self,
        model,
        output_dir: str,
        latent_creator,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.latent_creator = latent_creator
        self.step_wise_images = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        self._save_image(
            step=config.init_time_step,
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
            time_steps=None,
        )

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm,
    ) -> None:
        self._save_image(
            step=t + 1,
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
            time_steps=time_steps,
        )

    def call_interrupt(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm,
    ) -> None:
        self._save_composite(seed=seed)

    def _save_image(
        self,
        step: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm,
    ) -> None:
        unpack_latents = self.latent_creator.unpack_latents(latents=latents, height=config.height, width=config.width)
        # Flux2's unpack_latents returns 128-channel patchified latents that still need
        # decode_packed_latents (BN denorm + unpatchify → 32ch).  Ideogram4's unpack_latents
        # applies its own shift/scale and already returns 32-channel VAE-ready latents, so
        # calling decode_packed_latents on them causes a channel shape mismatch.
        vae_latent_channels = getattr(self.model.vae, "latent_channels", 32)
        if hasattr(self.model.vae, "decode_packed_latents") and unpack_latents.shape[1] > vae_latent_channels:
            stepwise_decoded = self.model.vae.decode_packed_latents(unpack_latents)
        else:
            stepwise_decoded = self.model.vae.decode(unpack_latents)
        generation_time = time_steps.format_dict["elapsed"] if time_steps is not None else 0
        stepwise_img = ImageUtil.to_image(
            decoded_latents=stepwise_decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.model.bits,
            lora_paths=self.model.lora_paths,
            lora_scales=self.model.lora_scales,
            generation_time=generation_time,
        )
        stepwise_img.save(
            path=self.output_dir / f"seed_{seed}_step{step}of{config.num_inference_steps}.png",
            export_json_metadata=False,
        )
        self.step_wise_images.append(stepwise_img)
        self._save_composite(seed=seed)

    def _save_composite(self, seed: int) -> None:
        if self.step_wise_images:
            composite_img = ImageUtil.to_composite_image(self.step_wise_images)
            composite_img.save(self.output_dir / f"seed_{seed}_composite.png")
