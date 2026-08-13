from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class ErnieWeightMapping(WeightMapping):
    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="adaln_modulation.weight",
                from_pattern=["adaLN_modulation.1.weight"],
            ),
            WeightTarget(
                to_pattern="adaln_modulation.bias",
                from_pattern=["adaLN_modulation.1.bias"],
            ),
            WeightTarget(
                to_pattern="x_embedder.proj.weight",
                from_pattern=["x_embedder.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="x_embedder.proj.bias",
                from_pattern=["x_embedder.proj.bias"],
            ),
            WeightTarget(
                to_pattern="text_proj.weight",
                from_pattern=["text_proj.weight"],
            ),
            WeightTarget(
                to_pattern="time_embedding.linear_1.weight",
                from_pattern=["time_embedding.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="time_embedding.linear_1.bias",
                from_pattern=["time_embedding.linear_1.bias"],
            ),
            WeightTarget(
                to_pattern="time_embedding.linear_2.weight",
                from_pattern=["time_embedding.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="time_embedding.linear_2.bias",
                from_pattern=["time_embedding.linear_2.bias"],
            ),
            WeightTarget(
                to_pattern="final_norm.linear.weight",
                from_pattern=["final_norm.linear.weight"],
            ),
            WeightTarget(
                to_pattern="final_norm.linear.bias",
                from_pattern=["final_norm.linear.bias"],
            ),
            WeightTarget(
                to_pattern="final_linear.weight",
                from_pattern=["final_linear.weight"],
            ),
            WeightTarget(
                to_pattern="final_linear.bias",
                from_pattern=["final_linear.bias"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.adaLN_sa_ln.weight",
                from_pattern=["layers.{layer}.adaLN_sa_ln.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.adaLN_mlp_ln.weight",
                from_pattern=["layers.{layer}.adaLN_mlp_ln.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.to_q.weight",
                from_pattern=["layers.{layer}.self_attention.to_q.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.to_k.weight",
                from_pattern=["layers.{layer}.self_attention.to_k.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.to_v.weight",
                from_pattern=["layers.{layer}.self_attention.to_v.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.to_out.0.weight",
                from_pattern=["layers.{layer}.self_attention.to_out.0.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.norm_q.weight",
                from_pattern=["layers.{layer}.self_attention.norm_q.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.norm_k.weight",
                from_pattern=["layers.{layer}.self_attention.norm_k.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.gate_proj.weight",
                from_pattern=["layers.{layer}.mlp.gate_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.up_proj.weight",
                from_pattern=["layers.{layer}.mlp.up_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.linear_fc2.weight",
                from_pattern=["layers.{layer}.mlp.linear_fc2.weight"],
            ),
        ]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        return Flux2WeightMapping.get_vae_mapping()
