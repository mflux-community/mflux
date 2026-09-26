from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class Qwen21LoRAMapping(LoRAMapping):
    """Key mapping for Qwen-Image-2.1 transformer LoRAs.

    PEFT adapters trained on the diffusers 2.1 transformer name their keys exactly like
    the mflux module tree (the base model keeps diffusers module names: attn.to_q/k/v,
    attn.to_out.0, img_mlp.*, modulation.1, time_text_embed.timestep_embedder.*), so the
    mapping below is 1:1 under three observed key prefixes: `transformer.` (PEFT-wrapped
    adapters, e.g. Viggle's Qwen-Image-2.1-viggle-turbo), `diffusion_model.` (ComfyUI
    exports), and bare module paths. RMSNorm/qk-norm layers are never LoRA targets.
    """

    _BLOCK_MODULES = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.gate_layer",
        "img_mlp.proj",
        "img_mlp.out",
    ]

    _GLOBAL_MODULES = [
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
    ]

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = []
        for module_path in Qwen21LoRAMapping._BLOCK_MODULES:
            targets.extend(Qwen21LoRAMapping._target(f"transformer_blocks.{{block}}.{module_path}"))
        for model_path in Qwen21LoRAMapping._GLOBAL_MODULES:
            targets.extend(Qwen21LoRAMapping._target(model_path))
        # MLX's nn.Sequential nests children under `.layers`: the shared modulation linear
        # (diffusers key `modulation.1`) lives at `modulation.layers.1` in the module tree.
        targets.extend(Qwen21LoRAMapping._target("modulation.layers.1", source_module="modulation.1"))
        return targets

    @staticmethod
    def _target(model_path: str, source_module: str | None = None) -> list[LoRATarget]:
        source_module = source_module or model_path
        up_patterns = []
        down_patterns = []
        for prefix in ("transformer", "diffusion_model", ""):
            base = f"{prefix}.{source_module}" if prefix else source_module
            up_patterns.extend([f"{base}.lora_B.weight", f"{base}.lora_B.default.weight"])
            down_patterns.extend([f"{base}.lora_A.weight", f"{base}.lora_A.default.weight"])
        return [
            LoRATarget(
                model_path=model_path,
                possible_up_patterns=up_patterns,
                possible_down_patterns=down_patterns,
            )
        ]
