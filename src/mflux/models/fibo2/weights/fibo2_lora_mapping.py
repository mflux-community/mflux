from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class Fibo2LoRAMapping:
    # Every linear layer, as diffusers' trainer adapts them; only the timestep MLP and final modulation are renamed
    PREFIXES = ("", "transformer.", "diffusion_model.", "base_model.model.")
    ATTENTION = ("to_q", "to_k", "to_v", "to_out.0")
    BLOCK = (
        *(f"attention.{name}" for name in ATTENTION),
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "adaLN_modulation.0",
    )
    INJECTION = tuple(f"cross_attention.{name}" for name in ATTENTION)
    PERCEIVER_LAYER = (
        *(f"cross_attention.{name}" for name in ATTENTION),
        *(f"self_attention.{name}" for name in ATTENTION),
        "ffn.w1",
        "ffn.w2",
        "ffn.w3",
    )

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = [
            Fibo2LoRAMapping._target("x_embedder"),
            Fibo2LoRAMapping._target("t_embedder.linear1", "t_embedder.mlp.0"),
            Fibo2LoRAMapping._target("t_embedder.linear2", "t_embedder.mlp.2"),
            Fibo2LoRAMapping._target("final_layer.linear"),
            Fibo2LoRAMapping._target("final_layer.adaLN_modulation.0", "final_layer.adaLN_modulation.1"),
        ]
        targets += [Fibo2LoRAMapping._target(f"noise_refiner.{{block}}.{name}") for name in Fibo2LoRAMapping.BLOCK]
        targets += [
            Fibo2LoRAMapping._target(f"layers.{{block}}.{name}")
            for name in Fibo2LoRAMapping.BLOCK + Fibo2LoRAMapping.INJECTION
        ]
        targets += [
            Fibo2LoRAMapping._target(f"perceiver.layers.{{block}}.{name}") for name in Fibo2LoRAMapping.PERCEIVER_LAYER
        ]
        return targets

    @staticmethod
    def _target(model_path: str, source_path: str | None = None) -> LoRATarget:
        source = source_path or model_path
        prefixes = Fibo2LoRAMapping.PREFIXES
        defaults = ("", ".default")
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=[f"{p}{source}.lora_B{d}.weight" for p in prefixes for d in defaults],
            possible_down_patterns=[f"{p}{source}.lora_A{d}.weight" for p in prefixes for d in defaults],
            possible_alpha_patterns=[f"{p}{source}.alpha" for p in prefixes],
        )
