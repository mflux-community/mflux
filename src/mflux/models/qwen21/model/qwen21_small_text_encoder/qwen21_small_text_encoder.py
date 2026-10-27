import inspect
import json

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from PIL import Image
from transformers import AutoTokenizer

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal
from mflux.models.common_models.qwen3_vl.qwen3_vl_processor import Qwen3VLProcessor
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen21_text_fusion import Qwen21TextFusion
from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen35_language_model import Qwen35LanguageModel
from mflux.models.qwen21.model.qwen21_small_text_encoder.zen_image_edit_weights import ZenImageEditWeights
from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition


class Qwen21SmallTextEncoder(nn.Module):
    # Qwen3.5-0.8B plus the AiArtLab/zen-image-edit text_fusion adapter, in place of Qwen3-VL-8B. It gives
    # the same (prompt embeds, image-slot mask) pair as the native encoder, after the system prefix.
    SYSTEM = "Comprehend and analyze the provided prompt."
    SYSTEM_PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    CLI_HELP = (
        "[EXPERIMENTAL] Replace the Qwen3-VL-8B text encoder with Qwen3.5-0.8B + the text-fusion adapter from "
        "AiArtLab/zen-image-edit. Downloads ~2.1 GB instead of ~17.5 GB and needs much less memory; English "
        "prompts only, and numerals in rendered text can come out wrong."
    )

    def __init__(self, encoder_config: dict, fusion_config: dict):
        super().__init__()
        text = encoder_config["text_config"]
        self.language_model = Qwen35LanguageModel(text)
        vision_config = encoder_config["vision_config"]
        if vision_config.get("deepstack_visual_indexes"):
            raise ValueError("The Qwen3.5 small text encoder does not support deepstack vision features.")
        vision = {
            key: value
            for key, value in vision_config.items()
            if key in inspect.signature(Qwen3VLVisionModel).parameters and key != "deepstack_visual_indexes"
        }
        self.visual = Qwen3VLVisionModel(**vision, preserve_input_dtype=True)
        self.visual.deepstack_visual_indexes = []
        self.visual.deepstack_merger_list = []
        Qwen3VLMultimodal.use_reference_activations(self.visual)
        self.text_fusion = Qwen21TextFusion(fusion_config)
        self.image_token_id = encoder_config["image_token_id"]
        self.merge_size = vision_config["spatial_merge_size"]
        self.student_layers = tuple(int(layer) for layer in fusion_config["student_layers"])
        self.drop_idx = int(fusion_config.get("drop_idx", 0))
        if text["hidden_size"] * len(self.student_layers) != fusion_config["in_dim"]:
            raise ValueError("text_fusion_config.in_dim does not match the encoder width and student_layers.")

    def encode(self, prompt: str, images: list[Image.Image], processor: Qwen3VLProcessor) -> tuple[mx.array, mx.array]:
        prefix = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(images)))
        text = (
            f"{Qwen21SmallTextEncoder.SYSTEM_PREFIX}"
            f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
        )
        rgb_images = []
        for image in images:
            rgba = image.convert("RGBA")
            white = Image.new("RGB", rgba.size, "white")
            white.paste(rgba, mask=rgba.getchannel("A"))
            rgb_images.append(white)
        inputs = processor(text=[text], return_tensors="np", **({"images": rgb_images} if images else {}))
        input_ids = mx.array(inputs["input_ids"])
        grids = mx.array(inputs["image_grid_thw"]) if images else None
        pixel_values = mx.array(inputs["pixel_values"]) if images else None
        embeds = self(input_ids, pixel_values, grids)
        slots = input_ids[0, self.drop_idx :] == self.image_token_id
        return embeds, slots

    def __call__(
        self, input_ids: mx.array, pixel_values: mx.array | None = None, image_grid_thw: mx.array | None = None
    ) -> mx.array:
        # The adapter reads the whole sequence (its position table is indexed by absolute token index),
        # so the system prefix is cut after the fusion, not before.
        fused = self.text_fusion(self.hidden_slices(input_ids, pixel_values, image_grid_thw))
        return fused[:, self.drop_idx :].astype(ModelConfig.precision)

    def hidden_slices(
        self, input_ids: mx.array, pixel_values: mx.array | None = None, image_grid_thw: mx.array | None = None
    ) -> mx.array:
        if input_ids.shape[0] != 1:
            raise ValueError("The Qwen3.5 small text encoder encodes one prompt at a time.")
        hidden = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_indices = mx.array(np.flatnonzero(np.asarray(input_ids[0]) == self.image_token_id), dtype=mx.int32)
            features, _ = self.visual(pixel_values.astype(mx.float32), image_grid_thw)
            if features.shape[0] != image_indices.size:
                raise ValueError("Vision feature count does not match the image placeholders.")
            hidden[:, image_indices] = features.astype(hidden.dtype)[None]
        positions = Qwen3VLMultimodal.mrope_positions(input_ids, image_grid_thw, self.image_token_id, self.merge_size)
        # (1, L, K * d): the student layers side by side, the layout the adapter was trained on.
        return mx.concatenate(self.language_model.hidden_states(hidden, positions, self.student_layers), axis=-1)

    @staticmethod
    def load() -> tuple["Qwen21SmallTextEncoder", Qwen3VLProcessor]:
        root = ZenImageEditWeights.snapshot()
        encoder_config = json.loads((root / "text_encoder" / "config.json").read_text())
        fusion_config = json.loads((root / "transformer" / "config.json").read_text())["text_fusion_config"]
        encoder = Qwen21SmallTextEncoder(encoder_config, fusion_config)
        processor = Qwen3VLProcessor(
            root / "processor", AutoTokenizer.from_pretrained(str(root / "processor"), local_files_only=True)
        )
        Qwen21SmallTextEncoder._check_drop_idx(processor, encoder.drop_idx)
        encoder.load_encoder_weights(mx.load(str(root / "text_encoder" / "model.safetensors")))
        Qwen21SmallTextEncoder.load_fusion_weights(encoder.text_fusion, ZenImageEditWeights.text_fusion_weights(root))
        return encoder, processor

    def load_encoder_weights(
        self, raw: dict[str, mx.array], language_model_dtype: mx.Dtype = ModelConfig.precision
    ) -> None:
        weights = {}
        for key, value in raw.items():
            if key.startswith("mtp.") or key.startswith("lm_head."):
                continue  # multi-token prediction and the (tied) LM head serve generation only
            key = key.removeprefix("model.")
            # The vision tower stays float32: its residual stream reaches ~2500 in the last block, and in
            # bfloat16 the image features drift to a cosine of ~0.76 against float32 (text is unaffected).
            target = mx.float32 if key.startswith("visual.") else language_model_dtype
            weights[key] = QwenImage21WeightDefinition.text_weight(key, value).astype(target)
        Qwen21SmallTextEncoder._strict_update(self, weights, exclude=("text_fusion.",), name="Qwen3.5-0.8B encoder")

    @staticmethod
    def load_fusion_weights(fusion: Qwen21TextFusion, raw: dict[str, mx.array]) -> None:
        weights = {}
        for key, value in raw.items():
            if key.endswith(".attn.in_proj_weight") or key.endswith(".attn.in_proj_bias"):
                # torch MultiheadAttention packs q, k and v into one projection.
                prefix, name = key.rsplit(".attn.", 1)
                suffix = "weight" if name == "in_proj_weight" else "bias"
                for projection, chunk in zip(("query_proj", "key_proj", "value_proj"), mx.split(value, 3, axis=0)):
                    weights[f"{prefix}.attn.{projection}.{suffix}"] = chunk.astype(mx.float32)
            else:
                weights[key] = value.astype(mx.float32)
        Qwen21SmallTextEncoder._strict_update(fusion, weights, name="text_fusion adapter")

    @staticmethod
    def _strict_update(
        module: nn.Module,
        weights: dict[str, mx.array],
        name: str,
        exclude: tuple[str, ...] = (),
    ) -> None:
        expected = {
            key: value
            for key, value in tree_flatten(module.parameters())
            if not key.startswith(exclude) and not key.endswith(".inv_freq")
        }
        missing = sorted(expected.keys() - weights.keys())
        unexpected = sorted(weights.keys() - expected.keys())
        mismatched = sorted(
            key for key in expected.keys() & weights.keys() if expected[key].shape != weights[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f"{name} weights do not fit: missing={missing[:5]}, unexpected={unexpected[:5]}, shapes={mismatched[:5]}"
            )
        module.load_weights(list(weights.items()), strict=False)

    @staticmethod
    def _check_drop_idx(processor: Qwen3VLProcessor, drop_idx: int) -> None:
        # The prefix is cut by position, so a tokenizer that disagrees would shift the whole condition.
        prefix = processor.tokenizer(Qwen21SmallTextEncoder.SYSTEM_PREFIX, add_special_tokens=False)["input_ids"]
        if len(prefix) != drop_idx:
            raise ValueError(
                f"text_fusion_config.drop_idx is {drop_idx}, but the system prefix is {len(prefix)} tokens."
            )
