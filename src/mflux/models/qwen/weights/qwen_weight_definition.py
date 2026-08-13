from typing import List

import mlx.core as mx

from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class QwenWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=QwenWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                mapping_getter=QwenWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="multi_json",
                precision=mx.bfloat16,
                mapping_getter=QwenWeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen",
                hf_subdir="tokenizer",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=1058,
                padding="longest",
                template="<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                download_patterns=["tokenizer/**", "added_tokens.json", "chat_template.jinja"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None):
        if not hasattr(module, "to_quantized"):
            return False

        if QwenWeightDefinition._is_text_encoder_path(path):
            return QwenWeightDefinition._text_encoder_quantization_predicate(path, module, bits)

        if bits == 4 and ".img_mod_linear" in path:
            return {"bits": 8}

        return True

    @staticmethod
    def quantization_predicate_for_loaded_weights(weights: LoadedWeights | None, bits: int | None):
        if bits in {4, 8}:
            if weights is None:
                return QwenWeightDefinition.quantization_predicate

            transformer_predicate = QwenWeightDefinition._transformer_quantization_predicate_for_loaded_weights(
                weights=weights,
                bits=bits,
            )
            text_encoder_predicate = QwenWeightDefinition._text_encoder_quantization_predicate_for_loaded_weights(
                weights=weights,
                bits=bits,
            )
            return QwenWeightDefinition._combined_quantization_predicate(
                transformer_predicate=transformer_predicate,
                text_encoder_predicate=text_encoder_predicate,
            )

        return QwenWeightDefinition._quantize_all_predicate

    @staticmethod
    def _transformer_quantization_predicate_for_loaded_weights(weights: LoadedWeights, bits: int | None):
        if bits != 4:
            return QwenWeightDefinition._quantize_all_predicate

        transformer = weights.components.get("transformer")
        if not isinstance(transformer, dict):
            return QwenWeightDefinition.quantization_predicate

        img_mod_bits = QwenWeightDefinition._quantized_linear_bits(
            transformer,
            "transformer_blocks.0.img_mod_linear",
        )
        if img_mod_bits == 8:
            return QwenWeightDefinition.quantization_predicate

        if img_mod_bits is None and QwenWeightDefinition._uses_unquantized_q4_sensitive_inputs(transformer):
            if QwenWeightDefinition._has_unquantized_txt_mod_linear(weights):
                return QwenWeightDefinition._post1_mixed_q4_quantization_predicate

            return QwenWeightDefinition._bf16_img_mod_mixed_q4_quantization_predicate

        return QwenWeightDefinition._quantize_all_predicate

    @staticmethod
    def _text_encoder_quantization_predicate_for_loaded_weights(weights: LoadedWeights, bits: int | None):
        text_encoder = weights.components.get("text_encoder")
        if not isinstance(text_encoder, dict):
            return QwenWeightDefinition._skip_text_encoder_quantization_predicate

        language_bits = QwenWeightDefinition._quantized_linear_bits(
            text_encoder,
            "encoder.layers.0.self_attn.q_proj",
        )
        vision_bits = QwenWeightDefinition._quantized_linear_bits(
            text_encoder,
            "encoder.visual.blocks.0.attn.qkv",
        )
        if language_bits is None and vision_bits is None:
            return QwenWeightDefinition._skip_text_encoder_quantization_predicate

        return QwenWeightDefinition._loaded_text_encoder_quantization_predicate(
            language_bits=language_bits,
            vision_bits=vision_bits,
            fallback_bits=bits,
        )

    @staticmethod
    def _combined_quantization_predicate(transformer_predicate, text_encoder_predicate):
        def predicate(path: str, module, bits: int | None = None):
            if QwenWeightDefinition._is_text_encoder_path(path):
                return text_encoder_predicate(path, module, bits)

            return transformer_predicate(path, module, bits)

        return predicate

    @staticmethod
    def _is_text_encoder_path(path: str) -> bool:
        return path.startswith("encoder.")

    @staticmethod
    def _is_text_encoder_vision_path(path: str) -> bool:
        return path.startswith("encoder.visual.")

    @staticmethod
    def _can_quantize_group64(module) -> bool:
        weight = getattr(module, "weight", None)
        shape = getattr(weight, "shape", None)
        return bool(shape and shape[-1] % 64 == 0)

    @staticmethod
    def _text_encoder_quantization_predicate(path: str, module, bits: int | None = None):
        if not hasattr(module, "to_quantized"):
            return False

        if not QwenWeightDefinition._can_quantize_group64(module):
            return False

        if bits == 4 and QwenWeightDefinition._is_text_encoder_vision_path(path):
            return {"bits": 8}

        return True

    @staticmethod
    def _loaded_text_encoder_quantization_predicate(
        language_bits: int | None,
        vision_bits: int | None,
        fallback_bits: int | None,
    ):
        def predicate(path: str, module, bits: int | None = None):
            if not hasattr(module, "to_quantized"):
                return False

            if not QwenWeightDefinition._can_quantize_group64(module):
                return False

            path_bits = vision_bits if QwenWeightDefinition._is_text_encoder_vision_path(path) else language_bits
            path_bits = path_bits or fallback_bits
            if path_bits == 8:
                return {"bits": 8}

            return path_bits is not None

        return predicate

    @staticmethod
    def _skip_text_encoder_quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        return False

    @staticmethod
    def _is_bf16_q4_sensitive_transformer_path(path: str) -> bool:
        if path in {"img_in", "txt_in", "norm_out.linear", "proj_out"}:
            return True

        if path.startswith("time_text_embed."):
            return True

        return ".img_mod_linear" in path

    @staticmethod
    def _uses_unquantized_q4_sensitive_inputs(transformer: dict) -> bool:
        return not QwenWeightDefinition._has_nested_key(transformer, "img_in.scales")

    @staticmethod
    def _has_nested_key(weights: dict, path: str) -> bool:
        current = weights
        for part in path.split("."):
            if isinstance(current, list):
                if not part.isdigit():
                    return False
                index = int(part)
                if index >= len(current):
                    return False
                current = current[index]
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        return True

    @staticmethod
    def _get_nested_value(weights: dict, path: str):
        current = weights
        for part in path.split("."):
            if isinstance(current, list):
                if not part.isdigit():
                    return None
                index = int(part)
                if index >= len(current):
                    return None
                current = current[index]
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @staticmethod
    def _quantized_linear_bits(weights: dict, path: str) -> int | None:
        weight = QwenWeightDefinition._get_nested_value(weights, f"{path}.weight")
        scales = QwenWeightDefinition._get_nested_value(weights, f"{path}.scales")
        weight_shape = getattr(weight, "shape", None)
        scales_shape = getattr(scales, "shape", None)
        if not weight_shape or not scales_shape:
            return None

        group_size = 64
        input_dims = scales_shape[-1] * group_size
        if input_dims == 0:
            return None

        bits = weight_shape[-1] * 32 / input_dims
        if bits in {4, 8}:
            return int(bits)

        return None

    @staticmethod
    def _has_unquantized_txt_mod_linear(weights: LoadedWeights | None) -> bool:
        if weights is None:
            return False

        transformer = weights.components.get("transformer")
        if not isinstance(transformer, dict):
            return False

        txt_mod_path = "transformer_blocks.0.txt_mod_linear"
        return QwenWeightDefinition._has_nested_key(
            transformer,
            f"{txt_mod_path}.weight",
        ) and not QwenWeightDefinition._has_nested_key(
            transformer,
            f"{txt_mod_path}.scales",
        )

    @staticmethod
    def _bf16_img_mod_mixed_q4_quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        if not hasattr(module, "to_quantized"):
            return False

        if bits == 4 and QwenWeightDefinition._is_bf16_q4_sensitive_transformer_path(path):
            return False

        return True

    @staticmethod
    def _post1_mixed_q4_quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        if not hasattr(module, "to_quantized"):
            return False

        if bits == 4 and (
            QwenWeightDefinition._is_bf16_q4_sensitive_transformer_path(path) or ".txt_mod_linear" in path
        ):
            return False

        return True

    @staticmethod
    def _quantize_all_predicate(path: str, module, bits: int | None = None) -> bool:
        return hasattr(module, "to_quantized")
        # The adaLN modulation producers are where 4-bit weight error compounds
        # across the denoising trajectory (upstream #484: flat-field sigma grows
        # 5.1 -> 16.1 from 4 to 50 steps at uniform q4, while q8 stays at the
        # bf16 floor). Keeping img_mod_linear at 8-bit restores that floor
        # (sigma 1.10/1.37 at 20/50 steps) for ~1.8 GB on the 4-bit model.
        # Same layer choice as upstream #420, so saves interoperate.
        if bits == 4 and ".img_mod_linear" in path:
            return {"bits": 8}
        return True
