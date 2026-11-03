from collections.abc import Callable

import mlx.core as mx

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget

PREFIXES = ("", "transformer.", "diffusion_model.", "base_model.model.")


class Flux3LoRAMapping(LoRAMapping):
    # MLX modules use the diffusers names. ComfyUI and ai-toolkit exports use the BFL names of the original
    # checkpoint (content_mode_blocks, single_blocks, q_proj, mlp_in, ...); both are accepted.
    _BLOCK_PARTS = [
        ("attn.to_q", "q_proj"),
        ("attn.to_k", "k_proj"),
        ("attn.to_v", "v_proj"),
        ("attn.to_out", "attn_out"),
        ("ff.linear_in", "mlp_in"),
        ("ff.linear_out", "mlp_out"),
    ]

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        targets.extend(Flux3LoRAMapping._get_global_targets())
        for mode in ("image", "image_cond"):
            targets.extend(Flux3LoRAMapping._get_block_targets(f"mode_blocks.{mode}", f"content_mode_blocks.{mode}"))
        targets.extend(Flux3LoRAMapping._get_block_targets("txt_mode_blocks", "txt_mode_blocks"))
        targets.extend(Flux3LoRAMapping._get_block_targets("single_transformer_blocks", "single_blocks"))
        return targets

    @staticmethod
    def _get_global_targets() -> list[LoRATarget]:
        targets = [
            Flux3LoRAMapping._target("context_embedder", aliases=["txt_in"]),
            Flux3LoRAMapping._target("timestep_embedder.linear_1", aliases=["time_in.in_layer"]),
            Flux3LoRAMapping._target("timestep_embedder.linear_2", aliases=["time_in.out_layer"]),
            Flux3LoRAMapping._target("target_timestep_embedder.linear_1", aliases=["ref_time_in.in_layer"]),
            Flux3LoRAMapping._target("target_timestep_embedder.linear_2", aliases=["ref_time_in.out_layer"]),
            Flux3LoRAMapping._target("proj_out", aliases=["final_layer.image.linear"]),
            Flux3LoRAMapping._target("norm_out.linear"),
            # BFL stores the final modulation as (shift, scale) and diffusers as (scale, shift), so the output
            # rows of a BFL-named update are swapped in halves.
            Flux3LoRAMapping._target(
                "norm_out.linear",
                aliases=["final_layer.image.adaLN_modulation.1"],
                include_model_path=False,
                up_transform=Flux3LoRAMapping._swap_halves,
            ),
        ]
        targets.extend(
            Flux3LoRAMapping._target(f"x_embedders.{mode}", aliases=[f"emb_in.{mode}"])
            for mode in ("image", "image_cond")
        )
        targets.extend(
            Flux3LoRAMapping._target(f"{stream}.{mode}.linear", aliases=[f"{stream}.{mode}.lin"])
            for stream in ("early_stream_modulations", "single_stream_modulations")
            for mode in ("image", "image_cond", "txt")
        )
        return targets

    @staticmethod
    def _get_block_targets(mlx: str, bfl: str) -> list[LoRATarget]:
        targets = []
        for part, bfl_part in Flux3LoRAMapping._BLOCK_PARTS:
            aliases = [f"{bfl}.{{block}}.{bfl_part}"]
            if part == "attn.to_out":
                aliases.append(f"{mlx}.{{block}}.attn.to_out.0")
            targets.append(Flux3LoRAMapping._target(f"{mlx}.{{block}}.{part}", aliases=aliases))
        # LoRAs extracted from the difference of two checkpoints (Base to distilled) also patch the q and k norm
        # scales directly (.diff). BFL names them norm.query_norm and norm.key_norm; the scales are the same values.
        for norm, bfl_norm in (("attn.norm_q", "norm.query_norm"), ("attn.norm_k", "norm.key_norm")):
            paths = [f"{mlx}.{{block}}.{norm}", f"{bfl}.{{block}}.{bfl_norm}"]
            targets.append(
                LoRATarget(
                    model_path=paths[0],
                    possible_up_patterns=[],
                    possible_down_patterns=[],
                    possible_diff_patterns=[f"{prefix}{path}.diff" for path in paths for prefix in PREFIXES],
                )
            )
        return targets

    @staticmethod
    def _target(
        model_path: str,
        aliases: list[str] | None = None,
        include_model_path: bool = True,
        up_transform: Callable[[mx.array], mx.array] | None = None,
    ) -> LoRATarget:
        module_paths = list(dict.fromkeys(([model_path] if include_model_path else []) + (aliases or [])))
        flat_paths = [path.replace(".", "_") for path in module_paths]
        # A LoKr update is a Kronecker product, so its rows cannot be swapped like a LoRA's: none is mapped there.
        lokr_paths = [] if up_transform else module_paths
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=Flux3LoRAMapping._matrix_patterns(module_paths, flat_paths, "up"),
            possible_down_patterns=Flux3LoRAMapping._matrix_patterns(module_paths, flat_paths, "down"),
            possible_alpha_patterns=Flux3LoRAMapping._suffix_patterns(module_paths, flat_paths, "alpha"),
            possible_lokr_w1_patterns=Flux3LoRAMapping._suffix_patterns(lokr_paths, [], "lokr_w1"),
            possible_lokr_w2_patterns=Flux3LoRAMapping._suffix_patterns(lokr_paths, [], "lokr_w2"),
            up_transform=up_transform,
        )

    @staticmethod
    def _matrix_patterns(module_paths: list[str], flat_paths: list[str], direction: str) -> list[str]:
        names = ("lora_B", "lora_up", "lora.up") if direction == "up" else ("lora_A", "lora_down", "lora.down")
        suffixes = [f"{name}{default}.weight" for name in names for default in ("", ".default")]
        patterns = [f"{prefix}{path}.{suffix}" for path in module_paths for prefix in PREFIXES for suffix in suffixes]
        flat = names[1]
        patterns += [f"lora_unet_{path}.{flat}{default}.weight" for path in flat_paths for default in ("", ".default")]
        return patterns

    @staticmethod
    def _suffix_patterns(module_paths: list[str], flat_paths: list[str], suffix: str) -> list[str]:
        patterns = [f"{prefix}{path}.{suffix}" for path in module_paths for prefix in PREFIXES]
        patterns += [f"lora_unet_{path}.{suffix}" for path in flat_paths]
        return patterns

    @staticmethod
    def _swap_halves(tensor: mx.array) -> mx.array:
        half = tensor.shape[0] // 2
        return mx.concatenate([tensor[half:], tensor[:half]], axis=0)
