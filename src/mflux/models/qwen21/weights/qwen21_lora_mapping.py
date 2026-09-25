import mlx.core as mx

from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class Qwen21LoRAMapping:
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = [
            LoRATarget(
                model_path=f"transformer_blocks.{{block}}.{name}",
                possible_up_patterns=[
                    f"transformer_blocks.{{block}}.{name}.lora_B.default.weight",
                    f"transformer.transformer_blocks.{{block}}.{name}.lora_B.weight",
                    f"diffusion_model.transformer_blocks.{{block}}.{name}.lora_B.weight",
                ],
                possible_down_patterns=[
                    f"transformer_blocks.{{block}}.{name}.lora_A.default.weight",
                    f"transformer.transformer_blocks.{{block}}.{name}.lora_A.weight",
                    f"diffusion_model.transformer_blocks.{{block}}.{name}.lora_A.weight",
                ],
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
        for name, transform in (
            ("img_mlp.gate_layer", Qwen21LoRAMapping._gate_up_gate),
            ("img_mlp.proj", Qwen21LoRAMapping._gate_up_proj),
        ):
            targets.append(
                LoRATarget(
                    model_path=f"transformer_blocks.{{block}}.{name}",
                    possible_up_patterns=[
                        f"{prefix}.transformer_blocks.{{block}}.img_mlp.gate_up.lora_B.weight"
                        for prefix in ("transformer", "diffusion_model")
                    ],
                    possible_down_patterns=[
                        f"{prefix}.transformer_blocks.{{block}}.img_mlp.gate_up.lora_A.weight"
                        for prefix in ("transformer", "diffusion_model")
                    ],
                    up_transform=transform,
                )
            )
        for source_path, model_path in (
            ("modulation.1", "modulation.layers.1"),
            ("time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_1"),
            ("time_text_embed.timestep_embedder.linear_2", "time_text_embed.timestep_embedder.linear_2"),
        ):
            targets.append(
                LoRATarget(
                    model_path=model_path,
                    possible_up_patterns=[
                        f"{prefix}.{source_path}.lora_B.weight" for prefix in ("transformer", "diffusion_model")
                    ],
                    possible_down_patterns=[
                        f"{prefix}.{source_path}.lora_A.weight" for prefix in ("transformer", "diffusion_model")
                    ],
                )
            )
        return targets

    @staticmethod
    def _gate_up_gate(tensor: mx.array) -> mx.array:
        return tensor[: tensor.shape[0] // 2]

    @staticmethod
    def _gate_up_proj(tensor: mx.array) -> mx.array:
        return tensor[tensor.shape[0] // 2 :]
