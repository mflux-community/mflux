from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class Qwen21LoRAMapping:
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        return [
            LoRATarget(
                model_path=f"transformer_blocks.{{block}}.{name}",
                possible_up_patterns=[f"transformer_blocks.{{block}}.{name}.lora_B.default.weight"],
                possible_down_patterns=[f"transformer_blocks.{{block}}.{name}.lora_A.default.weight"],
            )
            for name in (
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "img_mlp.proj",
                "img_mlp.gate_layer",
                "img_mlp.out",
            )
        ]
