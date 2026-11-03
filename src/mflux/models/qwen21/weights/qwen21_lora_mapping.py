import mlx.core as mx

from mflux.models.common.lora.mapping.lora_mapping import LoRATarget

PREFIXES = ("", "transformer.", "diffusion_model.", "base_model.model.")
UP_SUFFIXES = ("lora_B", "lora_B.default", "lora_up", "lora_up.default", "lora.up", "lora.up.default")
DOWN_SUFFIXES = ("lora_A", "lora_A.default", "lora_down", "lora_down.default", "lora.down", "lora.down.default")


class Qwen21LoRAMapping:
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = [
            Qwen21LoRAMapping._target(f"transformer_blocks.{{block}}.{name}")
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
                Qwen21LoRAMapping._target(
                    f"transformer_blocks.{{block}}.{name}",
                    source_path="transformer_blocks.{block}.img_mlp.gate_up",
                    up_transform=transform,
                )
            )
        for source_path, model_path in (
            ("modulation.1", "modulation.layers.1"),
            ("time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_1"),
            ("time_text_embed.timestep_embedder.linear_2", "time_text_embed.timestep_embedder.linear_2"),
            ("img_in", "img_in"),
            ("proj_out", "proj_out"),
            ("norm_out.linear", "norm_out.linear"),
            ("txt_in.in_layer", "txt_in.in_layer"),
            ("txt_in.out_layer", "txt_in.out_layer"),
        ):
            targets.append(Qwen21LoRAMapping._target(model_path, source_path=source_path))
        # ComfyUI turbo exports also ship direct .diff patches on the RMSNorm weights
        targets.extend(
            LoRATarget(
                model_path=path,
                possible_up_patterns=[],
                possible_down_patterns=[],
                possible_diff_patterns=[f"{p}{path}.diff" for p in PREFIXES],
            )
            for path in (
                "transformer_blocks.{block}.attn.norm_q",
                "transformer_blocks.{block}.attn.norm_k",
                "txt_in.text_norm",
            )
        )
        return targets

    @staticmethod
    def _target(model_path: str, source_path: str | None = None, up_transform=None) -> LoRATarget:
        # The same layer is named bare, transformer.- or diffusion_model.-prefixed, with lora_A/B, lora_down/up or
        # lora.down/up, and may carry a per-layer .alpha (ComfyUI and Lightning exports)
        source = source_path or model_path
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=[f"{p}{source}.{s}.weight" for p in PREFIXES for s in UP_SUFFIXES],
            possible_down_patterns=[f"{p}{source}.{s}.weight" for p in PREFIXES for s in DOWN_SUFFIXES],
            possible_alpha_patterns=[f"{p}{source}.alpha" for p in PREFIXES],
            up_transform=up_transform,
        )

    @staticmethod
    def _gate_up_gate(tensor: mx.array) -> mx.array:
        return tensor[: tensor.shape[0] // 2]

    @staticmethod
    def _gate_up_proj(tensor: mx.array) -> mx.array:
        return tensor[tensor.shape[0] // 2 :]
