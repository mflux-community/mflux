import pytest

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping


def _matched_targets(keys: list[str]) -> dict[str, set[str]]:
    patterns = LoRALoader._build_pattern_mappings(Qwen21LoRAMapping.get_mapping())
    targets: dict[str, set[str]] = {}
    for key in keys:
        hits = [m for m in patterns if LoRALoader._match_pattern(key, m.source_pattern) is not None]
        assert hits, f"key not matched by any pattern: {key}"
        assert len({h.matrix_name for h in hits}) == 1, f"key matches conflicting matrix names: {key}"
        assert len({h.target_path for h in hits}) == 1, f"key matches multiple targets: {key}"
        match = hits[0]
        block_idx = LoRALoader._match_pattern(key, match.source_pattern)
        target = match.target_path.format(block=block_idx) if "{block}" in match.target_path else match.target_path
        targets.setdefault(target, set()).add(match.matrix_name)
    return targets


@pytest.mark.fast
class TestQwen21LoRAMapping:
    def test_matches_peft_prefixed_block_keys(self):
        # Viggle's Qwen-Image-2.1-viggle-turbo ships PEFT-wrapped keys under transformer.
        keys = [
            "transformer.transformer_blocks.7.attn.to_q.lora_A.weight",
            "transformer.transformer_blocks.7.attn.to_q.lora_B.weight",
            "transformer.transformer_blocks.7.attn.to_k.lora_A.weight",
            "transformer.transformer_blocks.7.attn.to_v.lora_A.weight",
            "transformer.transformer_blocks.7.attn.to_out.0.lora_A.weight",
            "transformer.transformer_blocks.7.img_mlp.gate_layer.lora_A.weight",
            "transformer.transformer_blocks.7.img_mlp.proj.lora_B.weight",
            "transformer.transformer_blocks.7.img_mlp.out.lora_A.weight",
        ]
        targets = _matched_targets(keys)
        assert targets == {"transformer_blocks.7.attn.to_q": {"lora_A", "lora_B"}} | {
            "transformer_blocks.7.attn.to_k": {"lora_A"},
            "transformer_blocks.7.attn.to_v": {"lora_A"},
            "transformer_blocks.7.attn.to_out.0": {"lora_A"},
            "transformer_blocks.7.img_mlp.gate_layer": {"lora_A"},
            "transformer_blocks.7.img_mlp.proj": {"lora_B"},
            "transformer_blocks.7.img_mlp.out": {"lora_A"},
        }

    def test_maps_diffusers_modulation_to_mlx_sequential_path(self):
        # diffusers addresses the shared modulation linear as modulation.1; MLX's
        # nn.Sequential nests it under modulation.layers.1.
        keys = [
            "transformer.modulation.1.lora_A.weight",
            "transformer.modulation.1.lora_B.weight",
        ]
        assert _matched_targets(keys) == {"modulation.layers.1": {"lora_A", "lora_B"}}

    def test_matches_global_embedder_keys(self):
        keys = [
            "transformer.time_text_embed.timestep_embedder.linear_1.lora_A.weight",
            "transformer.time_text_embed.timestep_embedder.linear_2.lora_B.weight",
        ]
        targets = _matched_targets(keys)
        assert targets == {
            "time_text_embed.timestep_embedder.linear_1": {"lora_A"},
            "time_text_embed.timestep_embedder.linear_2": {"lora_B"},
        }

    def test_matches_comfyui_prefix_and_bare_keys(self):
        keys = [
            "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer_blocks.0.attn.to_q.lora_B.weight",
        ]
        targets = _matched_targets(keys)
        assert set(targets) == {"transformer_blocks.0.attn.to_q"}
        assert targets["transformer_blocks.0.attn.to_q"] == {"lora_A", "lora_B"}

    def test_matches_peft_default_adapter_suffix(self):
        keys = ["transformer.transformer_blocks.2.attn.to_v.lora_A.default.weight"]
        assert _matched_targets(keys) == {"transformer_blocks.2.attn.to_v": {"lora_A"}}

    def test_all_targets_resolve_on_real_transformer(self):
        # Every model_path in the mapping must walk to an existing nn.Linear.
        transformer = Qwen21Transformer()
        for target in Qwen21LoRAMapping.get_mapping():
            concrete = target.model_path.format(block=0)
            module = LoRALoader._get_target_module(transformer, concrete)
            assert hasattr(module, "weight"), f"target is not a linear layer: {concrete}"

    def test_full_viggle_style_key_set_matches(self):
        # 32 blocks x (7 block modules + globals) in the exact format of the shipped
        # adapter: every key matched, no conflicts.
        keys = []
        for i in range(32):
            for module in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "img_mlp.gate_layer",
                "img_mlp.proj",
                "img_mlp.out",
            ]:
                keys.extend(
                    [
                        f"transformer.transformer_blocks.{i}.{module}.lora_A.weight",
                        f"transformer.transformer_blocks.{i}.{module}.lora_B.weight",
                    ]
                )
        keys.extend(
            [
                "transformer.modulation.1.lora_A.weight",
                "transformer.modulation.1.lora_B.weight",
                "transformer.time_text_embed.timestep_embedder.linear_1.lora_A.weight",
                "transformer.time_text_embed.timestep_embedder.linear_1.lora_B.weight",
                "transformer.time_text_embed.timestep_embedder.linear_2.lora_A.weight",
                "transformer.time_text_embed.timestep_embedder.linear_2.lora_B.weight",
            ]
        )
        patterns = LoRALoader._build_pattern_mappings(Qwen21LoRAMapping.get_mapping())
        matched = {k for k in keys if any(LoRALoader._match_pattern(k, p.source_pattern) is not None for p in patterns)}
        assert matched == set(keys)
