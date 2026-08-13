from typing import List

import mlx.core as mx

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


class Ideogram4WeightMapping(WeightMapping):
    @staticmethod
    def prepare_tensor(key: str, value: mx.array) -> mx.array:
        if value.dtype == mx.uint8 or key.endswith(".weight_scale"):
            return value
        if value.dtype in (mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8, mx.bool_):
            return value
        return value.astype(ModelConfig.precision)

    @staticmethod
    def transform_text_encoder_key(key: str) -> str | None:
        if not key.startswith("language_model."):
            return None
        mapped = key[len("language_model.") :]
        if mapped.startswith(("embed_tokens.", "layers.", "norm.")):
            return mapped
        return None

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="input_proj.weight",
                from_pattern=["input_proj.weight"],
            ),
            WeightTarget(
                to_pattern="input_proj.weight_scale",
                from_pattern=["input_proj.weight_scale"],
            ),
            WeightTarget(
                to_pattern="input_proj.bias",
                from_pattern=["input_proj.bias"],
            ),
            WeightTarget(
                to_pattern="llm_cond_norm.weight",
                from_pattern=["llm_cond_norm.weight"],
            ),
            WeightTarget(
                to_pattern="llm_cond_proj.weight",
                from_pattern=["llm_cond_proj.weight"],
            ),
            WeightTarget(
                to_pattern="llm_cond_proj.weight_scale",
                from_pattern=["llm_cond_proj.weight_scale"],
            ),
            WeightTarget(
                to_pattern="llm_cond_proj.bias",
                from_pattern=["llm_cond_proj.bias"],
            ),
            WeightTarget(
                to_pattern="t_embedding.mlp_in.weight",
                from_pattern=["t_embedding.mlp_in.weight"],
            ),
            WeightTarget(
                to_pattern="t_embedding.mlp_in.weight_scale",
                from_pattern=["t_embedding.mlp_in.weight_scale"],
            ),
            WeightTarget(
                to_pattern="t_embedding.mlp_in.bias",
                from_pattern=["t_embedding.mlp_in.bias"],
            ),
            WeightTarget(
                to_pattern="t_embedding.mlp_out.weight",
                from_pattern=["t_embedding.mlp_out.weight"],
            ),
            WeightTarget(
                to_pattern="t_embedding.mlp_out.weight_scale",
                from_pattern=["t_embedding.mlp_out.weight_scale"],
            ),
            WeightTarget(
                to_pattern="t_embedding.mlp_out.bias",
                from_pattern=["t_embedding.mlp_out.bias"],
            ),
            WeightTarget(
                to_pattern="adaln_proj.weight",
                from_pattern=["adaln_proj.weight"],
            ),
            WeightTarget(
                to_pattern="adaln_proj.weight_scale",
                from_pattern=["adaln_proj.weight_scale"],
            ),
            WeightTarget(
                to_pattern="adaln_proj.bias",
                from_pattern=["adaln_proj.bias"],
            ),
            WeightTarget(
                to_pattern="embed_image_indicator.weight",
                from_pattern=["embed_image_indicator.weight"],
            ),
            WeightTarget(
                to_pattern="final_layer.adaln_modulation.weight",
                from_pattern=["final_layer.adaln_modulation.weight"],
            ),
            WeightTarget(
                to_pattern="final_layer.adaln_modulation.weight_scale",
                from_pattern=["final_layer.adaln_modulation.weight_scale"],
            ),
            WeightTarget(
                to_pattern="final_layer.adaln_modulation.bias",
                from_pattern=["final_layer.adaln_modulation.bias"],
            ),
            WeightTarget(
                to_pattern="final_layer.linear.weight",
                from_pattern=["final_layer.linear.weight"],
            ),
            WeightTarget(
                to_pattern="final_layer.linear.weight_scale",
                from_pattern=["final_layer.linear.weight_scale"],
            ),
            WeightTarget(
                to_pattern="final_layer.linear.bias",
                from_pattern=["final_layer.linear.bias"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.adaln_modulation.weight",
                from_pattern=["layers.{layer}.adaln_modulation.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.adaln_modulation.weight_scale",
                from_pattern=["layers.{layer}.adaln_modulation.weight_scale"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.adaln_modulation.bias",
                from_pattern=["layers.{layer}.adaln_modulation.bias"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.attention.qkv.weight",
                from_pattern=["layers.{layer}.attention.qkv.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.attention.qkv.weight_scale",
                from_pattern=["layers.{layer}.attention.qkv.weight_scale"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.attention.o.weight",
                from_pattern=["layers.{layer}.attention.o.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.attention.o.weight_scale",
                from_pattern=["layers.{layer}.attention.o.weight_scale"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.attention.norm_q.weight",
                from_pattern=["layers.{layer}.attention.norm_q.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.attention.norm_k.weight",
                from_pattern=["layers.{layer}.attention.norm_k.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.feed_forward.w1.weight",
                from_pattern=["layers.{layer}.feed_forward.w1.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.feed_forward.w1.weight_scale",
                from_pattern=["layers.{layer}.feed_forward.w1.weight_scale"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.feed_forward.w2.weight",
                from_pattern=["layers.{layer}.feed_forward.w2.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.feed_forward.w2.weight_scale",
                from_pattern=["layers.{layer}.feed_forward.w2.weight_scale"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.feed_forward.w3.weight",
                from_pattern=["layers.{layer}.feed_forward.w3.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.feed_forward.w3.weight_scale",
                from_pattern=["layers.{layer}.feed_forward.w3.weight_scale"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.attention_norm1.weight",
                from_pattern=["layers.{layer}.attention_norm1.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.ffn_norm1.weight",
                from_pattern=["layers.{layer}.ffn_norm1.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.attention_norm2.weight",
                from_pattern=["layers.{layer}.attention_norm2.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.ffn_norm2.weight",
                from_pattern=["layers.{layer}.ffn_norm2.weight"],
            ),
        ]
