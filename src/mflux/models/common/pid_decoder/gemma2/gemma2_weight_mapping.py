from typing import List

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget

# google/gemma-2-2b-it is a standard HF transformers checkpoint (config.json +
# model.safetensors, bf16), unrelated to the nvidia/PiD checkpoint that
# pid_weight_mapping.py handles. Key names verified against Gemma2Model /
# Gemma2Attention / Gemma2MLP / Gemma2TransformerBlock (2026-07-24): no biases
# on any Linear (matches HF's Gemma2), norm modules store the raw "weight"
# HF also stores (both apply `1.0 + weight` at forward time -- no transform needed).


class Gemma2WeightMapping:
    @staticmethod
    def get_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(to_pattern="embed_tokens.weight", from_pattern=["model.embed_tokens.weight"]),
            WeightTarget(to_pattern="norm.weight", from_pattern=["model.norm.weight"]),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.q_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.q_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.k_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.k_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.v_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.v_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.o_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.o_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.mlp.gate_proj.weight",
                from_pattern=["model.layers.{block}.mlp.gate_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.mlp.up_proj.weight",
                from_pattern=["model.layers.{block}.mlp.up_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.mlp.down_proj.weight",
                from_pattern=["model.layers.{block}.mlp.down_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.input_layernorm.weight",
                from_pattern=["model.layers.{block}.input_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.post_attention_layernorm.weight",
                from_pattern=["model.layers.{block}.post_attention_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.pre_feedforward_layernorm.weight",
                from_pattern=["model.layers.{block}.pre_feedforward_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.post_feedforward_layernorm.weight",
                from_pattern=["model.layers.{block}.post_feedforward_layernorm.weight"],
            ),
        ]


class Gemma2WeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="gemma2",
                hf_subdir="",
                num_blocks=26,  # google/gemma-2-2b-it: num_hidden_layers (Gemma2Config default)
                loading_mode="mlx_native",
                mapping_getter=Gemma2WeightMapping.get_mapping,
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return ["*.safetensors", "config.json"]
