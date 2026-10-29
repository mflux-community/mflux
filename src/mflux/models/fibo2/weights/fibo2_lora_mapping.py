from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer_block import Fibo2TransformerBlock


class Fibo2LoRAMapping:
    # Every linear layer, as diffusers' trainer adapts them; only the timestep MLP and final modulation are renamed
    PREFIXES = ("", "transformer.", "diffusion_model.", "base_model.model.")
    # Bria's feed-forward is 9536 wide; ComfyUI's repack (and ai-toolkit, which trains on it) zero-pads it to 9728, so
    # their LoRAs carry 192 extra rows per half of w13 and 192 extra columns of w2. Those are cut off here.
    FFN_HIDDEN = Fibo2TransformerBlock.ffn_hidden_dim(3584)
    # The MLX modules diffusers names differently, since it keeps them in an nn.Sequential
    DIFFUSERS_NAMES = {
        "t_embedder.linear1": "t_embedder.mlp.0",
        "t_embedder.linear2": "t_embedder.mlp.2",
        "final_layer.adaLN_modulation.0": "final_layer.adaLN_modulation.1",
    }
    TOP_LEVEL = (
        "x_embedder",
        "t_embedder.linear1",
        "t_embedder.linear2",
        "final_layer.linear",
        "final_layer.adaLN_modulation.0",
    )
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
        targets = [Fibo2LoRAMapping._target(name) for name in Fibo2LoRAMapping.TOP_LEVEL]
        targets += [Fibo2LoRAMapping._target(f"noise_refiner.{{block}}.{name}") for name in Fibo2LoRAMapping.BLOCK]
        targets += [
            Fibo2LoRAMapping._target(f"layers.{{block}}.{name}")
            for name in Fibo2LoRAMapping.BLOCK + Fibo2LoRAMapping.INJECTION
        ]
        targets += [
            Fibo2LoRAMapping._target(f"perceiver.layers.{{block}}.{name}") for name in Fibo2LoRAMapping.PERCEIVER_LAYER
        ]
        targets += Fibo2LoRAMapping._comfy_targets()
        targets += Fibo2LoRAMapping._norm_targets()
        return targets

    @staticmethod
    def diffusers_name(model_path: str) -> str:
        # The name diffusers, PEFT and ComfyUI give the module at this MLX path, which a saved LoRA's keys use
        return Fibo2LoRAMapping.DIFFUSERS_NAMES.get(model_path, model_path)

    @staticmethod
    def _norm_targets() -> list[LoRATarget]:
        # Extracted LoRAs (kohya, like Comfy-Org's turbo LoRA) also patch the norm weights directly (.diff). ComfyUI
        # names the self-attention q and k norms q_norm and k_norm; the other norms have the same names.
        names = [(f"{stack}.{norm}", f"{stack}.{norm}") for stack in ("noise_refiner.{block}", "layers.{block}") for norm in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2")]  # fmt: off
        names += [(f"{stack}.attention.norm_{x}", f"{stack}.attention.{x}_norm") for stack in ("noise_refiner.{block}", "layers.{block}") for x in "qk"]  # fmt: off
        names += [(f"layers.{{block}}.{norm}", f"layers.{{block}}.{norm}") for norm in ("cross_attn_norm1", "cross_attn_norm2", "cross_attention.norm_q", "cross_attention.norm_k")]  # fmt: off
        names += [(f"perceiver.layers.{{block}}.{norm}", f"perceiver.layers.{{block}}.{norm}") for norm in ("cross_attn_norm", "self_attn_norm", "ffn_norm", "cross_attention.norm_q", "cross_attention.norm_k", "self_attention.norm_q", "self_attention.norm_k")]  # fmt: off
        # final_layer.norm has no weight (LayerNorm without affine), so it takes no patch
        names += [("perceiver.norm_out", "perceiver.norm_out")]
        prefixes = Fibo2LoRAMapping.PREFIXES
        return [
            LoRATarget(
                model_path=path,
                possible_up_patterns=[],
                possible_down_patterns=[],
                possible_diff_patterns=[f"{p}{source}.diff" for p in prefixes],
            )  # fmt: off
            for path, source in names
        ]

    @staticmethod
    def _comfy_targets() -> list[LoRATarget]:
        # ComfyUI's layout (Comfy-Org/Fibo-2.0), which ai-toolkit also saves, fuses self-attention q, k and v into
        # attention.qkv, cross-attention k and v into to_kv, and the feed-forward gate and up into w13 (gate first),
        # and drops the .0 of to_out. A LoRA on a fused layer splits along its output: every part keeps the shared
        # down weight and takes its slice of the up weight.
        split = Fibo2LoRAMapping._split_targets
        targets = []
        for stack in ("noise_refiner.{block}", "layers.{block}"):
            targets += split(f"{stack}.attention.qkv", [f"{stack}.attention.to_{x}" for x in "qkv"])
            targets += split(
                f"{stack}.feed_forward.w13",
                [f"{stack}.feed_forward.w1", f"{stack}.feed_forward.w3"],
                Fibo2LoRAMapping.FFN_HIDDEN,
            )
            targets.append(Fibo2LoRAMapping._target(f"{stack}.attention.to_out.0", f"{stack}.attention.out"))
        for attention in ("layers.{block}.cross_attention", "perceiver.layers.{block}.cross_attention", "perceiver.layers.{block}.self_attention"):  # fmt: off
            targets += split(f"{attention}.to_kv", [f"{attention}.to_k", f"{attention}.to_v"])
            targets.append(Fibo2LoRAMapping._target(f"{attention}.to_out.0", f"{attention}.to_out"))
        targets += split("perceiver.layers.{block}.ffn.w13", ["perceiver.layers.{block}.ffn.w1", "perceiver.layers.{block}.ffn.w3"], Fibo2LoRAMapping.FFN_HIDDEN)  # fmt: off
        return targets

    @staticmethod
    def _split_targets(source_path: str, model_paths: list[str], keep: int | None = None) -> list[LoRATarget]:
        count = len(model_paths)
        return [
            Fibo2LoRAMapping._target(
                path, source_path, up_transform=lambda up, i=i: Fibo2LoRAMapping._part(up, i, count, keep)
            )  # fmt: off
            for i, path in enumerate(model_paths)
        ]

    @staticmethod
    def _part(up, index: int, count: int, keep: int | None = None):
        if up.shape[0] % count != 0:
            raise ValueError(f"A fused Fibo 2 LoRA up weight {tuple(up.shape)} does not split into {count} equal parts")
        rows = up.shape[0] // count
        return up[index * rows : (index + 1) * rows][:keep]

    @staticmethod
    def _unpad_down(down):
        # The input side of w2 is the feed-forward width: drop ComfyUI's zero padding, a no-op for unpadded LoRAs
        return down[:, : Fibo2LoRAMapping.FFN_HIDDEN]

    @staticmethod
    def _target(model_path: str, source_path: str | None = None, up_transform=None) -> LoRATarget:
        source = source_path or Fibo2LoRAMapping.diffusers_name(model_path)
        down_transform = Fibo2LoRAMapping._unpad_down if model_path.endswith(("feed_forward.w2", "ffn.w2")) else None
        prefixes = Fibo2LoRAMapping.PREFIXES
        defaults = ("", ".default")
        return LoRATarget(
            model_path=model_path,
            # PEFT names (lora_A/lora_B, as diffusers and ai-toolkit save) and kohya names (lora_down/lora_up)
            possible_up_patterns=[f"{p}{source}.lora_B{d}.weight" for p in prefixes for d in defaults]
            + [f"{p}{source}.lora_up.weight" for p in prefixes],  # fmt: off
            possible_down_patterns=[f"{p}{source}.lora_A{d}.weight" for p in prefixes for d in defaults]
            + [f"{p}{source}.lora_down.weight" for p in prefixes],  # fmt: off
            possible_alpha_patterns=[f"{p}{source}.alpha" for p in prefixes],
            possible_diff_b_patterns=[f"{p}{source}.diff_b" for p in prefixes],
            up_transform=up_transform,
            down_transform=down_transform,
        )
