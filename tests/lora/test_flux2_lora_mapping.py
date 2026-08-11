from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping


class TestFlux2LoRAMapping:
    def test_matches_diffusion_model_double_block_keys(self):
        keys = [
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight",
            "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight",
            "diffusion_model.double_blocks.0.txt_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight",
            "diffusion_model.double_blocks.0.txt_attn.proj.lora_A.weight",
            "diffusion_model.double_blocks.0.txt_attn.proj.lora_B.weight",
            "diffusion_model.double_blocks.0.img_mlp.0.lora_A.weight",
            "diffusion_model.double_blocks.0.img_mlp.0.lora_B.weight",
            "diffusion_model.double_blocks.0.img_mlp.2.lora_A.weight",
            "diffusion_model.double_blocks.0.img_mlp.2.lora_B.weight",
            "diffusion_model.double_blocks.0.txt_mlp.0.lora_A.weight",
            "diffusion_model.double_blocks.0.txt_mlp.0.lora_B.weight",
            "diffusion_model.double_blocks.0.txt_mlp.2.lora_A.weight",
            "diffusion_model.double_blocks.0.txt_mlp.2.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_lycoris_double_block_lokr_keys(self):
        keys = [
            "lycoris_double_blocks_0_img_attn_qkv.lokr_w1",
            "lycoris_double_blocks_0_img_attn_qkv.lokr_w2",
            "lycoris_double_blocks_0_img_attn_qkv.dora_scale",
            "lora_unet_double_blocks_0_img_attn_qkv.lokr_w1",
            "lora_unet_double_blocks_0_txt_mlp_2.dora_scale",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_diffusion_model_double_block_lokr_keys(self):
        keys = [
            "diffusion_model.double_blocks.0.img_attn.qkv.alpha",
            "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1",
            "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2",
            "diffusion_model.double_blocks.0.img_attn.proj.alpha",
            "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1",
            "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2",
            "diffusion_model.double_blocks.0.txt_attn.qkv.alpha",
            "diffusion_model.double_blocks.0.txt_attn.qkv.lokr_w1",
            "diffusion_model.double_blocks.0.txt_attn.qkv.lokr_w2",
            "diffusion_model.double_blocks.0.txt_attn.proj.alpha",
            "diffusion_model.double_blocks.0.txt_attn.proj.lokr_w1",
            "diffusion_model.double_blocks.0.txt_attn.proj.lokr_w2",
            "diffusion_model.double_blocks.0.img_mlp.0.alpha",
            "diffusion_model.double_blocks.0.img_mlp.0.lokr_w1",
            "diffusion_model.double_blocks.0.img_mlp.0.lokr_w2",
            "diffusion_model.double_blocks.0.img_mlp.2.alpha",
            "diffusion_model.double_blocks.0.img_mlp.2.lokr_w1",
            "diffusion_model.double_blocks.0.img_mlp.2.lokr_w2",
            "diffusion_model.double_blocks.0.txt_mlp.0.alpha",
            "diffusion_model.double_blocks.0.txt_mlp.0.lokr_w1",
            "diffusion_model.double_blocks.0.txt_mlp.0.lokr_w2",
            "diffusion_model.double_blocks.0.txt_mlp.2.alpha",
            "diffusion_model.double_blocks.0.txt_mlp.2.lokr_w1",
            "diffusion_model.double_blocks.0.txt_mlp.2.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_single_transformer_default_weight_keys(self):
        keys = [
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default.weight",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.default.weight",
            "single_transformer_blocks.0.attn.to_out.lora_A.default.weight",
            "single_transformer_blocks.0.attn.to_out.lora_B.default.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_single_transformer_lokr_keys(self):
        keys = [
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.alpha",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lokr_w1",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.lokr_w2",
            "single_transformer_blocks.0.attn.to_out.alpha",
            "single_transformer_blocks.0.attn.to_out.lokr_w1",
            "single_transformer_blocks.0.attn.to_out.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_lora_unet_klein_dora_scale_keys(self):
        keys = [
            "lora_unet_transformer_blocks_0_attn_to_q.dora_scale",
            "lora_unet_transformer_blocks_0_ff_linear_in.dora_scale",
            "lora_unet_single_transformer_blocks_0_attn_to_qkv_mlp_proj.dora_scale",
            "lora_unet_single_transformer_blocks_0_attn_to_out.dora_scale",
            "lora_unet_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w1",
            "lora_unet_single_transformer_blocks_0_attn_to_out.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_lycoris_single_transformer_lokr_keys(self):
        keys = [
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.alpha",
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.dora_scale",
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w1",
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w2",
            "lycoris_single_transformer_blocks_0_attn_to_out.alpha",
            "lycoris_single_transformer_blocks_0_attn_to_out.dora_scale",
            "lycoris_single_transformer_blocks_0_attn_to_out.lokr_w1",
            "lycoris_single_transformer_blocks_0_attn_to_out.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_transformer_block_default_weight_keys(self):
        keys = [
            "transformer_blocks.0.attn.to_q.lora_A.default.weight",
            "transformer_blocks.0.attn.to_q.lora_B.default.weight",
            "transformer_blocks.0.attn.to_out.0.lora_A.default.weight",
            "transformer_blocks.0.attn.to_out.0.lora_B.default.weight",
            "transformer_blocks.0.ff.linear_in.lora_A.default.weight",
            "transformer_blocks.0.ff.linear_in.lora_B.default.weight",
            "transformer_blocks.0.ff_context.linear_out.lora_A.default.weight",
            "transformer_blocks.0.ff_context.linear_out.lora_B.default.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_transformer_block_lokr_keys(self):
        keys = [
            "transformer_blocks.0.attn.to_q.alpha",
            "transformer_blocks.0.attn.to_q.lokr_w1",
            "transformer_blocks.0.attn.to_q.lokr_w2",
            "transformer_blocks.0.attn.to_out.0.alpha",
            "transformer_blocks.0.attn.to_out.0.lokr_w1",
            "transformer_blocks.0.attn.to_out.0.lokr_w2",
            "transformer_blocks.0.attn.add_q_proj.alpha",
            "transformer_blocks.0.attn.add_q_proj.lokr_w1",
            "transformer_blocks.0.attn.add_q_proj.lokr_w2",
            "transformer_blocks.0.ff.linear_in.alpha",
            "transformer_blocks.0.ff.linear_in.lokr_w1",
            "transformer_blocks.0.ff.linear_in.lokr_w2",
            "transformer_blocks.0.ff_context.linear_out.alpha",
            "transformer_blocks.0.ff_context.linear_out.lokr_w1",
            "transformer_blocks.0.ff_context.linear_out.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_factorized_lycoris_lokr_keys(self):
        keys = [
            "lycoris_transformer_blocks_0_ff_linear_in.lokr_w1_a",
            "lycoris_transformer_blocks_0_ff_linear_in.lokr_w1_b",
            "lycoris_transformer_blocks_0_ff_linear_in.lokr_w2_a",
            "lycoris_transformer_blocks_0_ff_linear_in.lokr_w2_b",
            "lycoris_transformer_blocks_0_ff_context_linear_out.lokr_t2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_lycoris_transformer_block_lokr_keys(self):
        keys = [
            "lycoris_transformer_blocks_0_attn_to_q.alpha",
            "lycoris_transformer_blocks_0_attn_to_q.dora_scale",
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w1",
            "lycoris_transformer_blocks_0_attn_to_q.lokr_w2",
            "lycoris_transformer_blocks_0_attn_to_out_0.alpha",
            "lycoris_transformer_blocks_0_attn_to_out_0.dora_scale",
            "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w1",
            "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w2",
            "lycoris_transformer_blocks_0_attn_add_q_proj.alpha",
            "lycoris_transformer_blocks_0_attn_add_q_proj.dora_scale",
            "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w1",
            "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w2",
            "lycoris_transformer_blocks_0_ff_linear_in.alpha",
            "lycoris_transformer_blocks_0_ff_linear_in.dora_scale",
            "lycoris_transformer_blocks_0_ff_linear_in.lokr_w1",
            "lycoris_transformer_blocks_0_ff_linear_in.lokr_w2",
            "lycoris_transformer_blocks_0_ff_context_linear_out.alpha",
            "lycoris_transformer_blocks_0_ff_context_linear_out.dora_scale",
            "lycoris_transformer_blocks_0_ff_context_linear_out.lokr_w1",
            "lycoris_transformer_blocks_0_ff_context_linear_out.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_diffusion_model_single_block_keys(self):
        keys = [
            "diffusion_model.single_blocks.0.linear1.lora_A.weight",
            "diffusion_model.single_blocks.0.linear1.lora_B.weight",
            "diffusion_model.single_blocks.0.linear2.lora_A.weight",
            "diffusion_model.single_blocks.0.linear2.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_diffusion_model_single_block_lokr_keys(self):
        keys = [
            "diffusion_model.single_blocks.0.linear1.lokr_w1",
            "diffusion_model.single_blocks.0.linear1.lokr_w2",
            "diffusion_model.single_blocks.0.linear2.lokr_w1",
            "diffusion_model.single_blocks.0.linear2.lokr_w2",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_preserves_existing_base_model_double_block_matches(self):
        keys = [
            "base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight",
            "base_model.model.double_blocks.0.txt_attn.proj.lora_A.weight",
            "base_model.model.double_blocks.0.txt_attn.proj.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def _matched_keys(self, keys: list[str]) -> set[str]:
        pattern_mappings = LoRALoader._build_pattern_mappings(Flux2LoRAMapping.get_mapping())
        matched_keys: set[str] = set()

        for key in keys:
            if any(LoRALoader._match_pattern(key, mapping.source_pattern) is not None for mapping in pattern_mappings):
                matched_keys.add(key)

        return matched_keys
