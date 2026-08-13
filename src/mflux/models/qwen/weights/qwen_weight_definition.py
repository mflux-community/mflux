from typing import List

import mlx.core as mx

from mflux.models.common.tokenizer import LanguageTokenizer
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
                skip_quantization=True,  # Quantization causes significant semantic degradation
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
        # The adaLN modulation producers are where 4-bit weight error compounds
        # across the denoising trajectory (upstream #484: flat-field sigma grows
        # 5.1 -> 16.1 from 4 to 50 steps at uniform q4, while q8 stays at the
        # bf16 floor). Keeping img_mod_linear at 8-bit restores that floor
        # (sigma 1.10/1.37 at 20/50 steps) for ~1.8 GB on the 4-bit model.
        # Same layer choice as upstream #420, so saves interoperate.
        if bits == 4 and ".img_mod_linear" in path:
            return {"bits": 8}
        return True
