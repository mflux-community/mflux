from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class Fibo2WeightMapping(WeightMapping):
    NUM_LAYERS = 44
    NUM_REFINER_LAYERS = 2
    NUM_PERCEIVER_LAYERS = 2
    ATTENTION = ("to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight", "norm_q.weight", "norm_k.weight")
    BLOCK = (
        "adaLN_modulation.0.weight",
        "adaLN_modulation.0.bias",
        "attention_norm1.weight",
        "attention_norm2.weight",
        "ffn_norm1.weight",
        "ffn_norm2.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
        *(f"attention.{name}" for name in ATTENTION),
    )
    INJECTION = (
        "cross_attn_gate",
        "cross_attn_norm1.weight",
        "cross_attn_norm2.weight",
        *(f"cross_attention.{name}" for name in ATTENTION),
    )
    PERCEIVER_LAYER = (
        "cross_attn_norm.weight",
        "self_attn_norm.weight",
        "ffn_norm.weight",
        "ffn.w1.weight",
        "ffn.w2.weight",
        "ffn.w3.weight",
        *(f"cross_attention.{name}" for name in ATTENTION),
        *(f"self_attention.{name}" for name in ATTENTION),
    )

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        renamed = [
            WeightTarget(to_pattern="t_embedder.linear1.weight", from_pattern=["t_embedder.mlp.0.weight"]),
            WeightTarget(to_pattern="t_embedder.linear1.bias", from_pattern=["t_embedder.mlp.0.bias"]),
            WeightTarget(to_pattern="t_embedder.linear2.weight", from_pattern=["t_embedder.mlp.2.weight"]),
            WeightTarget(to_pattern="t_embedder.linear2.bias", from_pattern=["t_embedder.mlp.2.bias"]),
            WeightTarget(
                to_pattern="final_layer.adaLN_modulation.0.weight",
                from_pattern=["final_layer.adaLN_modulation.1.weight"],
            ),
            WeightTarget(
                to_pattern="final_layer.adaLN_modulation.0.bias", from_pattern=["final_layer.adaLN_modulation.1.bias"]
            ),
        ]
        names = [
            "x_embedder.weight",
            "x_embedder.bias",
            "final_layer.linear.weight",
            "final_layer.linear.bias",
            "perceiver.latent_queries",
            "perceiver.norm_out.weight",
        ]
        same = [WeightTarget(to_pattern=name, from_pattern=[name]) for name in names]
        blocks = (
            Fibo2WeightMapping._blocks("noise_refiner", Fibo2WeightMapping.BLOCK, Fibo2WeightMapping.NUM_REFINER_LAYERS)
            + Fibo2WeightMapping._blocks("layers", Fibo2WeightMapping.BLOCK, Fibo2WeightMapping.NUM_LAYERS)
            + Fibo2WeightMapping._blocks("perceiver.layers", Fibo2WeightMapping.PERCEIVER_LAYER, Fibo2WeightMapping.NUM_PERCEIVER_LAYERS)
            # Only the injection layers have these
            + Fibo2WeightMapping._blocks("layers", Fibo2WeightMapping.INJECTION, Fibo2WeightMapping.NUM_LAYERS, required=False)
        )  # fmt: off
        return renamed + same + blocks

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        # The FIBO 2 VAE is the FLUX.2 VAE
        return Flux2WeightMapping.get_vae_mapping()

    @staticmethod
    def _blocks(prefix: str, names: tuple[str, ...], count: int, required: bool = True) -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern=f"{prefix}.{{block}}.{name}",
                from_pattern=[f"{prefix}.{{block}}.{name}"],
                max_blocks=count,
                required=required,
            )
            for name in names
        ]
