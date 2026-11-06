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
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.lora.path_util import expand_module_paths_from_targets
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler
from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import fused_qk_norm_rope_available
from mflux.models.qwen21.training_adapter.qwen21_comfy_lora import Qwen21ComfyLoRA
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping
from mflux.utils.image_util import ImageUtil
from mflux.utils.version_util import VersionUtil


class Qwen21TrainingAdapter(TrainingAdapter):
    """LoRA training for Qwen-Image-2.1 text to image, with ai-toolkit's recipe (qwen_image_2 in ostris/ai-toolkit).

    The noise level is sigma = sigmoid(N(SIGMA_MU, 1)): ai-toolkit's "sigmoid" timesteps shifted toward high noise, where
    the sampler spends most of its steps, with no resolution shift.
    The latent is noised to (1 - sigma) * clean + sigma * noise and the transformer, given sigma, predicts the velocity
    noise - clean. Captions are encoded as at inference (the last Qwen3-VL layer before its final norm, without the
    system turn), and CAPTION_DROPOUT of the steps get the empty caption, as ai-toolkit's caption_dropout_rate. The LoRA
    is saved with ai-toolkit's keys (Qwen21ComfyLoRA), which ComfyUI and mflux --lora read.
    """

    # ai-toolkit's caption_dropout_rate for this model
    CAPTION_DROPOUT = 0.05
    # Mean of the logit-normal noise level: sigmoid(N(SIGMA_MU, 1)), median sigma 0.73. With ai-toolkit's 0 (median 0.5),
    # a 300-step style LoRA learned the look but was too weak at scale 1 (blueprint-blue share 0.004 to 0.062 over
    # 3 seeds); with 1.0 it gave 0.90 to 0.92, and a character LoRA kept its identity
    SIGMA_MU = 1.0

    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        if Qwen21TurboScheduler.is_turbo(model_config):
            raise ValueError('Qwen-Image-2.1 LoRAs are trained on the base model: set "model": "qwen-image-2.1" (Turbo is distilled).')  # fmt: off
        self._model_config = model_config
        self._qwen = QwenImage21(quantize=quantize, model_config=model_config, model_path=model_path, bake_lora=False)
        self._guidance = 1.0
        self._empty: dict[str, mx.array] | None = None
        # The preview prompts' encodings once the text encoder is gone (release_text_encoder)
        self._preview_encodings: dict[str, tuple[mx.array, mx.array]] = {}

    def model(self):
        return self._qwen

    def transformer(self):
        return self._qwen.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        # The schedule is for the previews; training draws its own noise levels (sample_sigma)
        self._guidance = training_spec.guidance or 1.0
        return Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=self._guidance,
        )

    def freeze_base(self) -> None:
        self._qwen.vae.freeze()
        self._qwen.transformer.freeze()
        if self._qwen.text_encoder is not None:  # gone after release_text_encoder
            self._qwen.text_encoder.freeze()
        Qwen21TrainingAdapter._set_fused_prologue(self._qwen.transformer, False)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self._qwen.transformer.gradient_checkpointing = enabled

    def sample_sigma(self, *, rng: random.Random, image_seq_len: int) -> float:  # noqa: ARG002
        return 1.0 / (1.0 + math.exp(-(rng.gauss(0.0, 1.0) + Qwen21TrainingAdapter.SIGMA_MU)))

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
            raise ValueError("Qwen-Image-2.1 training supports text to image only.")
        image = ImageUtil.scale_to_dimensions(ImageUtil.load_image(image_path).convert("RGB"), width, height)
        encoded = VAEUtil.encode(vae=self._qwen.vae, image=ImageUtil.to_array(image), tiling_config=self._qwen.tiling_config)  # fmt: off
        # Qwen21LatentCreator.pack_latents, kept in float32 for the loss target
        clean = mx.reshape(encoded, (1, 64, height // 16, width // 16))
        clean = mx.reshape(mx.transpose(clean, (0, 2, 3, 1)), (1, -1, 64)).astype(mx.float32)
        embeds, mask = self._qwen._encode_prompt(prompt)
        self._empty_caption()
        self._qwen.prompt_cache = {}
        cond = {"embeds": embeds, "mask": mask}
        mx.eval(clean, cond)
        return clean, cond

    def training_condition(self, *, cond: Any, rng: random.Random) -> Any:
        if rng.random() < Qwen21TrainingAdapter.CAPTION_DROPOUT:
            return self._empty_caption()
        return cond

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:
        velocity = self._qwen.transformer.forward_training(
            hidden_states=latents_t.astype(ModelConfig.precision),
            encoder_hidden_states=cond["embeds"],
            sigma=sigmas[t : t + 1],
            height=config.height,
            width=config.width,
            encoder_hidden_states_mask=cond["mask"],
        )
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
        # After release_text_encoder, the prompts come from the encodings kept then. No negative prompt, so no CFG,
        # as mflux-generate-qwen-2.1 samples by default
        self._qwen.prompt_cache = dict(self._preview_encodings)
        Qwen21TrainingAdapter._set_fused_prologue(self._qwen.transformer, True)
        try:
            with TrainingUtil.assistant_disabled(self._qwen.transformer):
                image = self._qwen.generate_image(
                    seed=seed,
                    prompt=prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=self._guidance,
                )
        finally:
            Qwen21TrainingAdapter._set_fused_prologue(self._qwen.transformer, False)
            # The cache lives on the module, so its encodings would become trainable parameters of model()
            self._qwen.prompt_cache = {}
        return image.image

    def release_text_encoder(self, training_spec: TrainingSpec) -> bool:
        # Encodes the preview prompts and the empty caption now, keeps them on the adapter, and drops the 16 GB Qwen3-VL
        # text encoder, which training never runs again
        monitoring = training_spec.monitoring
        prompts = list(monitoring.preview_prompts) if monitoring is not None else []
        self._empty_caption()  # the data may come from a cache, without encode_data
        for prompt in prompts:
            self._preview_encodings[prompt] = self._qwen._encode_prompt(prompt)
        mx.eval(list(self._preview_encodings.values()))
        self._qwen.prompt_cache = {}
        self._qwen.text_encoder = None
        return True

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:
        loras = {}
        for module_path, _, _ in expand_module_paths_from_targets(training_spec.lora_layers.targets):
            lora = TrainingUtil.get_train_lora(self._qwen.transformer, module_path)
            loras[module_path] = (mx.transpose(lora.lora_A), mx.transpose(lora.lora_B), float(lora.scale))
        mx.save_safetensors(
            str(path),
            Qwen21ComfyLoRA.weights(loras),
            metadata={
                "format": "pt",
                "mflux_version": VersionUtil.get_mflux_version(),
                "model": training_spec.model,
                Qwen21ComfyLoRA.RANKS_KEY: json.dumps({p: int(down.shape[0]) for p, (down, _, _) in loras.items()}),
                Qwen21ComfyLoRA.SCALES_KEY: json.dumps({p: scale for p, (_, _, scale) in loras.items()}),
            },
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        # Resuming: the fused gate_up would give gate_layer and proj the stacked down weight at twice the rank, so a
        # run's own checkpoint is split back into its modules, which the saved optimizer state matches
        weights, metadata = mx.load(str(path), return_metadata=True)
        ranks = metadata.get(Qwen21ComfyLoRA.RANKS_KEY)
        if ranks is None:
            self._load(path, role="train", scale=1.0)
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            split_path = Path(temp_dir) / "adapter.safetensors"
            split = Qwen21ComfyLoRA.split(weights, json.loads(ranks), json.loads(metadata[Qwen21ComfyLoRA.SCALES_KEY]))
            mx.save_safetensors(str(split_path), split)
            self._load(split_path, role="train", scale=1.0)

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        self._load(path, role="assistant", scale=scale)

    def _load(self, path: str | Path, *, role: str, scale: float) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Qwen21LoRAMapping.get_mapping(),
            transformer=self._qwen.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role=role,
            bake_lora=False,
        )

    def _empty_caption(self) -> dict[str, mx.array]:
        # Encoded once, while the text encoder is there (encode_data runs before release_text_encoder)
        if self._empty is None:
            embeds, mask = self._qwen._encode_prompt("")
            self._empty = {"embeds": embeds, "mask": mask}
            mx.eval(self._empty)
            self._qwen.prompt_cache = {}
        return self._empty

    @staticmethod
    def _set_fused_prologue(transformer, enabled: bool) -> None:
        # The fused Q/K norm + RoPE kernel has no gradient: training uses the composed ops, previews the kernel
        for block in transformer.transformer_blocks:
            block.attn.use_fused_prologue = enabled and Qwen21TrainingAdapter._fused_available(block.attn)

    @staticmethod
    def _fused_available(attention) -> bool:
        return fused_qk_norm_rope_available(attention.head_dim) and attention.norm_q.eps == 1e-6
