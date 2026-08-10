from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping


class TestFluxLoRAMapping:
    def test_matches_factorized_lycoris_lokr_keys(self):
        keys = [
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w1_a",
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w1_b",
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w2_a",
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w2_b",
            "lycoris_transformer_blocks_0_attn_to_q.lokr_t2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_observed_lycoris_single_transformer_lokr_keys(self):
        keys = [
            "lycoris_single_transformer_blocks_0_attn_to_q.alpha",
            "lycoris_single_transformer_blocks_0_attn_to_q.lokr_w1",
            "lycoris_single_transformer_blocks_0_attn_to_q.lokr_w2",
            "lycoris_single_transformer_blocks_0_proj_mlp.alpha",
            "lycoris_single_transformer_blocks_0_proj_mlp.lokr_w1",
            "lycoris_single_transformer_blocks_0_proj_mlp.lokr_w2",
            "lycoris_single_transformer_blocks_0_norm_linear.alpha",
            "lycoris_single_transformer_blocks_0_norm_linear.lokr_w1",
            "lycoris_single_transformer_blocks_0_norm_linear.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_observed_double_underscore_lycoris_lokr_keys(self):
        keys = [
            "lycoris__x_embedder.alpha",
            "lycoris__x_embedder.lokr_w1",
            "lycoris__x_embedder.lokr_w2",
            "lycoris__context_embedder.alpha",
            "lycoris__context_embedder.lokr_w1",
            "lycoris__context_embedder.lokr_w2",
            "lycoris__time_text_embed_guidance_embedder_linear_1.alpha",
            "lycoris__time_text_embed_guidance_embedder_linear_1.lokr_w1",
            "lycoris__time_text_embed_guidance_embedder_linear_1.lokr_w2",
            "lycoris__norm_out_linear.alpha",
            "lycoris__norm_out_linear.lokr_w1",
            "lycoris__norm_out_linear.lokr_w2",
            "lycoris__single_transformer_blocks_0_proj_out.alpha",
            "lycoris__single_transformer_blocks_0_proj_out.lokr_w1",
            "lycoris__single_transformer_blocks_0_proj_out.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_lycoris_dora_scale_and_lora_unet_lokr_keys(self):
        keys = [
            "lycoris_transformer_blocks_0_attn_to_q.dora_scale",
            "lora_unet_transformer_blocks_0_attn_to_q.dora_scale",
            "lora_unet_transformer_blocks_0_attn_to_q.lokr_w1",
            "lora_unet_transformer_blocks_0_attn_to_q.lokr_w2",
            "lycoris_single_transformer_blocks_0_attn_to_q.dora_scale",
            "lora_unet_single_transformer_blocks_0_attn_to_q.lokr_w1",
            "lora_unet_single_transformer_blocks_0_attn_to_q.lokr_w2",
            "lycoris__x_embedder.dora_scale",
            "lora_unet_x_embedder.dora_scale",
            "lora_unet_x_embedder.lokr_w1",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_observed_lycoris_transformer_lokr_keys(self):
        keys = [
            "lycoris_transformer_blocks_0_attn_to_q.alpha",
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w1",
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w2",
            "lycoris_transformer_blocks_0_attn_to_out_0.alpha",
            "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w1",
            "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w2",
            "lycoris_transformer_blocks_0_ff_net_0_proj.alpha",
            "lycoris_transformer_blocks_0_ff_net_0_proj.lokr_w1",
            "lycoris_transformer_blocks_0_ff_net_0_proj.lokr_w2",
            "lycoris_transformer_blocks_0_ff_context_net_2.alpha",
            "lycoris_transformer_blocks_0_ff_context_net_2.lokr_w1",
            "lycoris_transformer_blocks_0_ff_context_net_2.lokr_w2",
            "lycoris_transformer_blocks_0_norm1_linear.alpha",
            "lycoris_transformer_blocks_0_norm1_linear.lokr_w1",
            "lycoris_transformer_blocks_0_norm1_linear.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def _matched_keys(self, keys: list[str]) -> set[str]:
        pattern_mappings = LoRALoader._build_pattern_mappings(FluxLoRAMapping.get_mapping())
        matched_keys: set[str] = set()

        for key in keys:
            if any(LoRALoader._match_pattern(key, mapping.source_pattern) is not None for mapping in pattern_mappings):
                matched_keys.add(key)

        return matched_keys
