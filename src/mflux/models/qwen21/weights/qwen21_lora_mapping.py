from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class Qwen21LoRAMapping:
    @staticmethod
    def _lora_target(model_path: str, *source_paths: str) -> LoRATarget:
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=[
                f"{path}.lora_B{adapter_name}.weight" for path in source_paths for adapter_name in (".default", "")
            ]
            + [f"{path}.lora_up" for path in source_paths],
            possible_down_patterns=[
                f"{path}.lora_A{adapter_name}.weight" for path in source_paths for adapter_name in (".default", "")
            ]
            + [f"{path}.lora_down" for path in source_paths],
        )

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = [
            Qwen21LoRAMapping._lora_target(
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
                Qwen21LoRAMapping._lora_target(
                    "time_text_embed.timestep_embedder.linear_1",
                    "time_text_embed.timestep_embedder.linear_1",
                    "transformer.time_text_embed.timestep_embedder.linear_1",
                ),
                Qwen21LoRAMapping._lora_target(
                    "time_text_embed.timestep_embedder.linear_2",
                    "time_text_embed.timestep_embedder.linear_2",
                    "transformer.time_text_embed.timestep_embedder.linear_2",
                ),
                Qwen21LoRAMapping._lora_target(
                    "modulation.layers.1",
                    "modulation.1",
                    "transformer.modulation.1",
                ),
                Qwen21LoRAMapping._lora_target("img_in", "img_in", "transformer.img_in"),
                Qwen21LoRAMapping._lora_target("norm_out.linear", "norm_out.linear", "transformer.norm_out.linear"),
                Qwen21LoRAMapping._lora_target("txt_in.in_layer", "txt_in.in_layer", "transformer.txt_in.in_layer"),
                Qwen21LoRAMapping._lora_target("txt_in.out_layer", "txt_in.out_layer", "transformer.txt_in.out_layer"),
            ]
        )
        return targets
