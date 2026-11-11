from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class SenseNovaLoRAMapping:
    # The official LoRA names the gen-stream linears of the checkpoint (language_model.model.layers.N.self_attn.q_proj_mot_gen,
    # ...mlp_mot_gen.gate_proj) with kohya matrices and a per-layer alpha. The transformer's modules drop the _mot_gen suffix.
    PREFIXES = ("", "diffusion_model.", "transformer.", "base_model.model.")
    SOURCE_ROOT = "language_model.model.layers.{block}"
    UP_SUFFIXES = ("lora_up", "lora_B")
    DOWN_SUFFIXES = ("lora_down", "lora_A")
    WEIGHT_SUFFIXES = ("", ".weight", ".default.weight")

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        root = SenseNovaLoRAMapping.SOURCE_ROOT
        attention = [(f"self_attn.{p}_proj", f"{root}.self_attn.{p}_proj_mot_gen") for p in "qkvo"]
        mlp = [(f"mlp.{p}_proj", f"{root}.mlp_mot_gen.{p}_proj") for p in ("gate", "up", "down")]
        return [SenseNovaLoRAMapping._target(f"layers.{{block}}.{path}", source) for path, source in attention + mlp]

    @staticmethod
    def _target(model_path: str, source: str) -> LoRATarget:
        names = [f"{prefix}{name}" for prefix in SenseNovaLoRAMapping.PREFIXES for name in (source, source.removeprefix("language_model.model."))]  # fmt: skip
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=SenseNovaLoRAMapping._matrix(names, SenseNovaLoRAMapping.UP_SUFFIXES),
            possible_down_patterns=SenseNovaLoRAMapping._matrix(names, SenseNovaLoRAMapping.DOWN_SUFFIXES),
            possible_alpha_patterns=[f"{name}.alpha" for name in names],
        )

    @staticmethod
    def _matrix(names: list[str], matrices: tuple[str, ...]) -> list[str]:
        return [f"{name}.{matrix}{suffix}" for name in names for matrix in matrices for suffix in SenseNovaLoRAMapping.WEIGHT_SUFFIXES]  # fmt: skip
