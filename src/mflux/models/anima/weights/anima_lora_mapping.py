from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class AnimaLoRAMapping(LoRAMapping):
    # The transformer's module paths are the checkpoint's, so a LoRA names them directly: kohya's sd-scripts
    # (networks.lora_anima) flattens them to lora_unet_blocks_0_self_attn_q_proj, PEFT-style exports keep the dots
    # under a diffusion_model. or similar prefix. The LLM adapter is covered too, for LoRAs trained with it.
    PREFIXES = ("", "diffusion_model.", "model.diffusion_model.", "net.", "transformer.", "base_model.model.")

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        return [AnimaLoRAMapping._target(path) for path in AnimaLoRAMapping.module_paths()]

    @staticmethod
    def module_paths() -> list[str]:
        block = "blocks.{block}"
        adapter = "llm_adapter.blocks.{block}"
        return [
            "x_embedder.proj.1",
            "t_embedder.1.linear_1",
            "t_embedder.1.linear_2",
            "final_layer.adaln_modulation.1",
            "final_layer.adaln_modulation.2",
            "final_layer.linear",
            *(
                f"{block}.{attn}.{proj}"
                for attn in ("self_attn", "cross_attn")
                for proj in ("q_proj", "k_proj", "v_proj", "output_proj")
            ),
            f"{block}.mlp.layer1",
            f"{block}.mlp.layer2",
            *(
                f"{block}.adaln_modulation_{name}.{index}"
                for name in ("self_attn", "cross_attn", "mlp")
                for index in (1, 2)
            ),
            *(
                f"{adapter}.{attn}.{proj}"
                for attn in ("self_attn", "cross_attn")
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
            ),
            f"{adapter}.mlp.0",
            f"{adapter}.mlp.2",
            "llm_adapter.out_proj",
        ]

    @staticmethod
    def _target(path: str) -> LoRATarget:
        dotted = [f"{prefix}{path}" for prefix in AnimaLoRAMapping.PREFIXES]
        flat = [f"{prefix}_{path.replace('.', '_')}" for prefix in ("lora_unet", "lycoris")]
        return LoRATarget(
            model_path=path,
            possible_up_patterns=AnimaLoRAMapping._matrix(dotted, flat, ("lora_B", "lora_up", "lora.up"), "lora_up"),
            possible_down_patterns=AnimaLoRAMapping._matrix(
                dotted, flat, ("lora_A", "lora_down", "lora.down"), "lora_down"
            ),
            possible_alpha_patterns=[f"{name}.alpha" for name in dotted + flat],
            possible_lokr_w1_patterns=[f"{name}.lokr_w1" for name in dotted + flat],
            possible_lokr_w2_patterns=[f"{name}.lokr_w2" for name in dotted + flat],
        )

    @staticmethod
    def _matrix(dotted: list[str], flat: list[str], names: tuple[str, ...], flat_name: str) -> list[str]:
        suffixes = ("weight", "default.weight")
        patterns = [f"{path}.{name}.{suffix}" for path in dotted for name in names for suffix in suffixes]
        return patterns + [f"{path}.{flat_name}.{suffix}" for path in flat for suffix in suffixes]
