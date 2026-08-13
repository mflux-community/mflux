from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping


def _identity(pattern: str, max_blocks: int | None = None) -> WeightTarget:
    """Identity weight target (MLX key == checkpoint key).

    ``max_blocks`` pins how many ``{block}`` indices to expand; required because
    the loader otherwise defaults to 4, and Boogu's block-stack families have
    different lengths.
    """
    return WeightTarget(to_pattern=pattern, from_pattern=[pattern], max_blocks=max_blocks)


# Boogu-Image-Turbo block-stack sizes (num_refiner_layers=2, num_double_stream_layers=8,
# num_layers=40 -> 32 single-stream; Qwen3-VL text decoder = 36 layers).
_NUM_REFINER_LAYERS = 2
_NUM_DOUBLE_STREAM_LAYERS = 8
_NUM_SINGLE_STREAM_LAYERS = 32
_NUM_TEXT_ENCODER_LAYERS = 36


# Per-block patterns shared by the modulated base blocks (noise/ref refiners,
# single-stream layers). ``{block}`` expands over the layer indices present.
_MODULATED_BLOCK_SUFFIXES = [
    "attn.norm_q.weight",
    "attn.norm_k.weight",
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "feed_forward.linear_1.weight",
    "feed_forward.linear_2.weight",
    "feed_forward.linear_3.weight",
    "ffn_norm1.weight",
    "ffn_norm2.weight",
    "norm1.linear.weight",
    "norm1.linear.bias",
    "norm1.norm.weight",
    "norm2.weight",
]

# Context refiner is non-modulated: norm1 is a plain RMSNorm (weight only).
_CONTEXT_BLOCK_SUFFIXES = [
    "attn.norm_q.weight",
    "attn.norm_k.weight",
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "feed_forward.linear_1.weight",
    "feed_forward.linear_2.weight",
    "feed_forward.linear_3.weight",
    "ffn_norm1.weight",
    "ffn_norm2.weight",
    "norm1.weight",
    "norm2.weight",
]

_DOUBLE_STREAM_SUFFIXES = [
    # Joint image<->instruction attention (host norms + out, processor projections).
    "img_instruct_attn.norm_q.weight",
    "img_instruct_attn.norm_k.weight",
    "img_instruct_attn.to_out.0.weight",
    "img_instruct_attn.processor.img_to_q.weight",
    "img_instruct_attn.processor.img_to_k.weight",
    "img_instruct_attn.processor.img_to_v.weight",
    "img_instruct_attn.processor.instruct_to_q.weight",
    "img_instruct_attn.processor.instruct_to_k.weight",
    "img_instruct_attn.processor.instruct_to_v.weight",
    "img_instruct_attn.processor.instruct_out.weight",
    "img_instruct_attn.processor.img_out.weight",
    # Image self-attention.
    "img_self_attn.norm_q.weight",
    "img_self_attn.norm_k.weight",
    "img_self_attn.to_q.weight",
    "img_self_attn.to_k.weight",
    "img_self_attn.to_v.weight",
    "img_self_attn.to_out.0.weight",
    # Feed-forwards.
    "img_feed_forward.linear_1.weight",
    "img_feed_forward.linear_2.weight",
    "img_feed_forward.linear_3.weight",
    "instruct_feed_forward.linear_1.weight",
    "instruct_feed_forward.linear_2.weight",
    "instruct_feed_forward.linear_3.weight",
    # Modulation norms.
    "img_norm1.linear.weight",
    "img_norm1.linear.bias",
    "img_norm1.norm.weight",
    "img_norm2.linear.weight",
    "img_norm2.linear.bias",
    "img_norm2.norm.weight",
    "img_norm3.linear.weight",
    "img_norm3.linear.bias",
    "img_norm3.norm.weight",
    "instruct_norm1.linear.weight",
    "instruct_norm1.linear.bias",
    "instruct_norm1.norm.weight",
    "instruct_norm2.linear.weight",
    "instruct_norm2.linear.bias",
    "instruct_norm2.norm.weight",
    # Plain RMSNorms.
    "img_ffn_norm1.weight",
    "img_ffn_norm2.weight",
    "img_attn_norm.weight",
    "img_self_attn_norm.weight",
    "instruct_ffn_norm1.weight",
    "instruct_ffn_norm2.weight",
    "instruct_attn_norm.weight",
]

_SINGLETONS = [
    "x_embedder.weight",
    "x_embedder.bias",
    "ref_image_patch_embedder.weight",
    "ref_image_patch_embedder.bias",
    "norm_out.linear_1.weight",
    "norm_out.linear_1.bias",
    "norm_out.linear_2.weight",
    "norm_out.linear_2.bias",
    "time_caption_embed.caption_embedder.0.weight",
    "time_caption_embed.caption_embedder.1.weight",
    "time_caption_embed.caption_embedder.1.bias",
    "time_caption_embed.timestep_embedder.linear_1.weight",
    "time_caption_embed.timestep_embedder.linear_1.bias",
    "time_caption_embed.timestep_embedder.linear_2.weight",
    "time_caption_embed.timestep_embedder.linear_2.bias",
    "image_index_embedding",
]


class BooguWeightMapping(WeightMapping):
    """Boogu-Image weight mappings (transformer identity, mllm prefix-strip, FLUX VAE)."""

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        targets: List[WeightTarget] = [_identity(p) for p in _SINGLETONS]
        for family, suffixes, count in (
            ("noise_refiner", _MODULATED_BLOCK_SUFFIXES, _NUM_REFINER_LAYERS),
            ("ref_image_refiner", _MODULATED_BLOCK_SUFFIXES, _NUM_REFINER_LAYERS),
            ("context_refiner", _CONTEXT_BLOCK_SUFFIXES, _NUM_REFINER_LAYERS),
            ("single_stream_layers", _MODULATED_BLOCK_SUFFIXES, _NUM_SINGLE_STREAM_LAYERS),
            ("double_stream_layers", _DOUBLE_STREAM_SUFFIXES, _NUM_DOUBLE_STREAM_LAYERS),
        ):
            targets.extend(_identity(f"{family}.{{block}}.{suffix}", count) for suffix in suffixes)
        return targets

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        """Qwen3-VL text decoder: strip the ``model.language_model.`` prefix."""
        prefix = "model.language_model."
        targets = [
            WeightTarget(to_pattern="embed_tokens.weight", from_pattern=[f"{prefix}embed_tokens.weight"]),
            WeightTarget(to_pattern="norm.weight", from_pattern=[f"{prefix}norm.weight"]),
        ]
        layer_suffixes = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        targets.extend(
            WeightTarget(
                to_pattern=f"layers.{{block}}.{suffix}",
                from_pattern=[f"{prefix}layers.{{block}}.{suffix}"],
                max_blocks=_NUM_TEXT_ENCODER_LAYERS,
            )
            for suffix in layer_suffixes
        )
        return targets

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """Boogu ships the FLUX.1 VAE — reuse the FLUX v1 VAE mapping verbatim."""
        return FluxWeightMapping.get_vae_mapping()
