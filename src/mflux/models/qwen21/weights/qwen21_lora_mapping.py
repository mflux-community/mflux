from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class Qwen21LoRAMapping:
    @staticmethod
    def _peft_target(model_path: str, *source_paths: str) -> LoRATarget:
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=[
                f"{path}.lora_B{adapter_name}.weight" for path in source_paths for adapter_name in (".default", "")
            ],
            possible_down_patterns=[
                f"{path}.lora_A{adapter_name}.weight" for path in source_paths for adapter_name in (".default", "")
            ],
        )

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = [
            Qwen21LoRAMapping._peft_target(
                f"transformer_blocks.{{block}}.{name}",
                f"transformer_blocks.{{block}}.{name}",
                f"transformer.transformer_blocks.{{block}}.{name}",
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
        targets.extend(
            [
                Qwen21LoRAMapping._peft_target(
                    "time_text_embed.timestep_embedder.linear_1",
                    "time_text_embed.timestep_embedder.linear_1",
                    "transformer.time_text_embed.timestep_embedder.linear_1",
                ),
                Qwen21LoRAMapping._peft_target(
                    "time_text_embed.timestep_embedder.linear_2",
                    "time_text_embed.timestep_embedder.linear_2",
                    "transformer.time_text_embed.timestep_embedder.linear_2",
                ),
                Qwen21LoRAMapping._peft_target(
                    "modulation.layers.1",
                    "modulation.1",
                    "transformer.modulation.1",
                ),
            ]
        )
        return targets
