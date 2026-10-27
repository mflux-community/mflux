from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class AnimaWeightMapping:
    # The transformer modules carry the checkpoint's own names, so its mapping is the identity. Every block is
    # listed by index: a checkpoint with fewer blocks is rejected at load instead of leaving random weights.
    NUM_BLOCKS = 28
    NUM_ADAPTER_BLOCKS = 6
    NUM_TEXT_ENCODER_LAYERS = 28
    # ComfyUI saves the transformer under model.diffusion_model. (Turbo, Aesthetic); the training format of the
    # base model uses net.
    TRANSFORMER_PREFIXES = ("model.diffusion_model.", "net.", "diffusion_model.")
    TOP_LEVEL = [
        "x_embedder.proj.1.weight",
        "t_embedder.1.linear_1.weight",
        "t_embedder.1.linear_2.weight",
        "t_embedding_norm.weight",
        "final_layer.adaln_modulation.1.weight",
        "final_layer.adaln_modulation.2.weight",
        "final_layer.linear.weight",
        "llm_adapter.embed.weight",
        "llm_adapter.out_proj.weight",
        "llm_adapter.out_proj.bias",
        "llm_adapter.norm.weight",
    ]

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        names = list(AnimaWeightMapping.TOP_LEVEL)
        for block in range(AnimaWeightMapping.NUM_BLOCKS):
            names += [f"blocks.{block}.{suffix}" for suffix in AnimaWeightMapping._block_suffixes()]
        for block in range(AnimaWeightMapping.NUM_ADAPTER_BLOCKS):
            names += [f"llm_adapter.blocks.{block}.{suffix}" for suffix in AnimaWeightMapping._adapter_block_suffixes()]  # fmt: off
        return [WeightTarget(to_pattern=name, from_pattern=[name]) for name in names]

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        # Qwen3-0.6B-Base in the transformers layout.
        return Flux2WeightMapping.get_text_encoder_mapping()

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        # The Qwen-Image VAE; the ComfyUI file is read through QwenWeightMapping.vae_key_from_comfy.
        return QwenWeightMapping.get_vae_mapping()

    @staticmethod
    def transformer_key(key: str) -> str:
        for prefix in AnimaWeightMapping.TRANSFORMER_PREFIXES:
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    @staticmethod
    def _attention_suffixes(attention: str, output: str) -> List[str]:
        return [f"{attention}.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", output, "q_norm", "k_norm")]

    @staticmethod
    def _block_suffixes() -> List[str]:
        modulations = ("adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp")
        return [
            *AnimaWeightMapping._attention_suffixes("self_attn", "output_proj"),
            *AnimaWeightMapping._attention_suffixes("cross_attn", "output_proj"),
            "mlp.layer1.weight",
            "mlp.layer2.weight",
            *(f"{modulation}.{index}.weight" for modulation in modulations for index in (1, 2)),
        ]

    @staticmethod
    def _adapter_block_suffixes() -> List[str]:
        return [
            *AnimaWeightMapping._attention_suffixes("self_attn", "o_proj"),
            *AnimaWeightMapping._attention_suffixes("cross_attn", "o_proj"),
            "norm_self_attn.weight",
            "norm_cross_attn.weight",
            "norm_mlp.weight",
            *(f"mlp.{index}.{param}" for index in (0, 2) for param in ("weight", "bias")),
        ]
