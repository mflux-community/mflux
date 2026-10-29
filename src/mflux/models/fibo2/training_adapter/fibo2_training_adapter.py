from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.lora.path_util import expand_module_paths_from_targets
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.model.fibo2_text_encoder.prompt_encoder import Fibo2PromptEncoder
from mflux.models.fibo2.training_adapter.fibo2_comfy_lora import Fibo2ComfyLoRA
from mflux.models.fibo2.variants.fibo2_core import _Fibo2Core
from mflux.models.fibo2.variants.txt2img.fibo2 import Fibo2
from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping
from mflux.utils.image_util import ImageUtil
from mflux.utils.version_util import VersionUtil


class Fibo2TrainingAdapter(TrainingAdapter):
    """LoRA training for FIBO 2 text to image, with the recipe of Bria's trainer (fine_tuning/train_lora.py in
    briaai/Fibo-2.0-dev).

    The noise level is sigma = sigmoid(N(shift, 1)) with probability 0.9 and U(0, 1) otherwise. The shift grows
    linearly with the image's token count, from 0.5 at 256 tokens to 1.15 at 4096: the mu of the inference schedule.
    The trainer noises the latent to (1 - sigma) * clean + sigma * noise, and the transformer, given sigma, predicts the
    velocity noise - clean, which is the loss target. Captions are encoded as at inference: a JSON caption cleaned and
    serialized compactly, a short description as it is, and a caption with text_render entries with the images of its
    text, and without them above noise level 0.9. The LoRA is saved in ComfyUI's layout (Fibo2ComfyLoRA): diffusion_model.*
    keys, fused q/k/v, k/v and w13, alpha / rank folded into the up weights, and each module's rank in the metadata so a
    resumed run gets its own layers back.
    """

    UNIFORM_SHARE = 0.1

    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        self._model_config = model_config
        self._fibo2 = Fibo2(quantize=quantize, model_config=model_config, model_path=model_path)
        self._guidance: float | None = None

    def model(self):
        return self._fibo2

    def transformer(self):
        return self._fibo2.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        self._guidance = training_spec.guidance
        config = Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=training_spec.guidance,
            scheduler="flow_match_euler_discrete",
        )
        config.scheduler.set_mu(FlowMatchEulerDiscreteScheduler._compute_linear_mu(config.image_seq_len))
        return config

    def freeze_base(self) -> None:
        self._fibo2.vae.freeze()
        self._fibo2.transformer.freeze()
        if self._fibo2.text_encoder is not None:  # gone after release_text_encoder
            self._fibo2.text_encoder.freeze()

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self._fibo2.transformer.gradient_checkpointing = enabled

    def sample_sigma(self, *, rng: random.Random, image_seq_len: int) -> float:
        if rng.random() < Fibo2TrainingAdapter.UNIFORM_SHARE:
            return rng.random()
        shift = FlowMatchEulerDiscreteScheduler._compute_linear_mu(image_seq_len)
        return 1.0 / (1.0 + math.exp(-(rng.gauss(0.0, 1.0) + shift)))

    def encode_data(
        self,
        *,
        data_id: int,  # noqa: ARG002
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,
    ) -> tuple[mx.array, Any]:
        if input_image_path is not None:
            raise ValueError("FIBO 2 training supports text to image only.")
        image = ImageUtil.scale_to_dimensions(ImageUtil.load_image(image_path).convert("RGB"), width, height)
        # As the edit path encodes an image, but kept in float32 for the loss target
        pixels = ImageUtil.to_array(image).astype(ModelConfig.precision)
        encoded = VAEUtil.encode(self._fibo2.vae, pixels, tiling_config=self._fibo2.tiling_config)
        clean_latents = self._fibo2.vae.normalize_latents(Fibo2LatentCreator.pack_latents(encoded))
        embeds, base_embeds = self._fibo2._encode_prompts(prompt, None, None, render_text_images=True)
        self._clear_prompt_cache()
        cond = {"embeds": embeds} if base_embeds is None else {"embeds": embeds, "base_embeds": base_embeds}
        mx.eval(clean_latents, cond)
        return clean_latents, cond

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:  # noqa: ARG002
        sigma = sigmas[t : t + 1]
        embeds = _Fibo2Core._embeds_for_step(float(sigma[0]), cond["embeds"], cond.get("base_embeds"))
        velocity = self._fibo2.transformer(latents_t, sigma, embeds)
        return velocity.astype(latents_t.dtype)

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,  # noqa: ARG002
    ):
        # After release_text_encoder, the prompts come from the encodings kept then
        self._fibo2.prompt_cache = dict(getattr(self, "_preview_embeds", {}))
        with TrainingUtil.assistant_disabled(self._fibo2.transformer):
            image = self._fibo2.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance,
            )
        self._clear_prompt_cache()
        return image.image

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:
        # Saved in ComfyUI's layout, which ComfyUI and ai-toolkit read and mflux loads back (Fibo2LoRAMapping)
        loras = {}
        for module_path, _, _ in expand_module_paths_from_targets(training_spec.lora_layers.targets):
            lora = TrainingUtil.get_train_lora(self._fibo2.transformer, module_path)
            loras[module_path] = (mx.transpose(lora.lora_A), mx.transpose(lora.lora_B), float(lora.scale))
        ranks = {module_path: int(down.shape[0]) for module_path, (down, _, _) in loras.items()}
        scales = {module_path: scale for module_path, (_, _, scale) in loras.items()}
        mx.save_safetensors(
            str(path),
            Fibo2ComfyLoRA.weights(loras),
            metadata={
                "format": "pt",
                "mflux_version": VersionUtil.get_mflux_version(),
                "model": training_spec.model,
                Fibo2ComfyLoRA.RANKS_KEY: json.dumps(ranks),
                Fibo2ComfyLoRA.SCALES_KEY: json.dumps(scales),
            },
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        # Resuming: a fused layer loaded as is would give every part the whole stacked down weight, so a run's own
        # checkpoint is split back into its modules at their ranks and scales, which the saved optimizer state matches
        weights, metadata = mx.load(str(path), return_metadata=True)
        ranks = metadata.get(Fibo2ComfyLoRA.RANKS_KEY)
        if ranks is None:
            self._load_train_lora(path)
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            split_path = Path(temp_dir) / "adapter.safetensors"
            scales = metadata.get(Fibo2ComfyLoRA.SCALES_KEY)
            split = Fibo2ComfyLoRA.split(weights, json.loads(ranks), json.loads(scales) if scales else None)
            mx.save_safetensors(str(split_path), split)
            self._load_train_lora(split_path)

    def _load_train_lora(self, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Fibo2LoRAMapping.get_mapping(),
            transformer=self._fibo2.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
            bake_lora=False,
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Fibo2LoRAMapping.get_mapping(),
            transformer=self._fibo2.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
            bake_lora=False,
        )

    def release_text_encoder(self, training_spec: TrainingSpec) -> bool:
        # Encodes the preview prompts (and the default negative when guidance is on) now, keeps them on the adapter, and
        # drops the text encoder. A preview caption that draws text images needs the encoder at every run, so then it
        # stays. The encodings live here, not in the model's prompt cache, which training would treat as parameters.
        monitoring = training_spec.monitoring
        prompts = list(monitoring.preview_prompts) if monitoring is not None else []
        if any(
            self._fibo2._text_images(Fibo2PromptEncoder.caption(Fibo2PromptEncoder.format_prompt(prompt)))
            for prompt in prompts
        ):
            return False
        for prompt in prompts:
            self._fibo2._encode_prompts(prompt, None, None, render_text_images=True)
        if training_spec.guidance is not None and training_spec.guidance > 1.0:
            self._fibo2._encode_prompt(Fibo2PromptEncoder.TEXT_TO_IMAGE_NEGATIVE)
        self._preview_embeds = dict(self._fibo2.prompt_cache)
        mx.eval(list(self._preview_embeds.values()))
        self._clear_prompt_cache()
        self._fibo2.text_encoder = None
        return True

    def _clear_prompt_cache(self) -> None:
        # The model is a module, so arrays left in its prompt cache would be trained and decayed with the LoRA
        self._fibo2.prompt_cache = {}
