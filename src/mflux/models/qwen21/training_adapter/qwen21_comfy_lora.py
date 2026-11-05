import mlx.core as mx


class Qwen21ComfyLoRA:
    # Writes trained Qwen-Image-2.1 LoRAs with the keys ai-toolkit saves and ComfyUI reads: diffusion_model.* names, and
    # the feed-forward gate and up in one img_mlp.gate_up layer, as in the Comfy-Org repack. mflux trains gate_layer and
    # proj as two layers, so the fused layer stacks them exactly: the two down weights one above the other, the two up
    # weights block-diagonal (gate rows first), and a rank of twice the trained one. Each layer's alpha / rank is folded
    # into its up weight. Qwen21LoRAMapping splits gate_up back into gate_layer and proj.
    FUSED = (".img_mlp.gate_layer", ".img_mlp.proj")
    GATE_UP = ".img_mlp.gate_up"
    # Metadata keys with each trained module's rank and scale, so a resumed run gets its own layers back
    RANKS_KEY = "mflux_lora_ranks"
    SCALES_KEY = "mflux_lora_scales"
    # MLX module paths whose checkpoint name differs (Qwen21LoRAMapping reads the checkpoint names)
    CHECKPOINT_NAMES = {"modulation.layers.1": "modulation.1"}

    @staticmethod
    def check_targets(paths: list[str]) -> None:
        # Runs before the model loads, so a target set the file cannot hold stops before training instead of at the
        # first checkpoint
        trained = set(paths)
        for path in paths:
            prefix = Qwen21ComfyLoRA._block_prefix(Qwen21ComfyLoRA._gate_of(path))
            if prefix is None:
                continue
            gate, proj = (prefix + suffix for suffix in Qwen21ComfyLoRA.FUSED)
            if gate not in trained or proj not in trained:
                raise ValueError(f"{gate} and {proj} train together (ComfyUI's img_mlp.gate_up): add both or neither.")

    @staticmethod
    def weights(loras: dict[str, tuple[mx.array, mx.array, float]]) -> dict[str, mx.array]:
        # loras: MLX module path -> (down (rank, in), up (out, rank), scale)
        out = {}
        for path, (down, up, scale) in loras.items():
            prefix = Qwen21ComfyLoRA._block_prefix(Qwen21ComfyLoRA._gate_of(path))
            if prefix is None:
                name = Qwen21ComfyLoRA.CHECKPOINT_NAMES.get(path, path)
                out[f"diffusion_model.{name}.lora_A.weight"] = down
                out[f"diffusion_model.{name}.lora_B.weight"] = up * scale
                continue
            gate, proj = (prefix + suffix for suffix in Qwen21ComfyLoRA.FUSED)
            if gate not in loras or proj not in loras:
                raise ValueError(f"{gate} and {proj} train together (ComfyUI's img_mlp.gate_up): add both or neither.")
            if path == proj:
                continue  # written with its gate_layer
            proj_down, proj_up, proj_scale = loras[proj]
            stacked = mx.zeros((up.shape[0] + proj_up.shape[0], down.shape[0] + proj_down.shape[0]), dtype=up.dtype)
            stacked[: up.shape[0], : down.shape[0]] = up * scale
            stacked[up.shape[0] :, down.shape[0] :] = proj_up * proj_scale
            out[f"diffusion_model.{prefix}{Qwen21ComfyLoRA.GATE_UP}.lora_A.weight"] = mx.concatenate([down, proj_down])
            out[f"diffusion_model.{prefix}{Qwen21ComfyLoRA.GATE_UP}.lora_B.weight"] = stacked
        return out

    @staticmethod
    def split(weights: dict[str, mx.array], ranks: dict[str, int], scales: dict[str, float]) -> dict[str, mx.array]:
        # The inverse of weights() for a run's own checkpoint: each module's down and up at its own rank, with alpha
        # / rank taken out of up again and written as .alpha, under the names Qwen21LoRAMapping reads as unfused layers
        out = {}
        for path, rank in ranks.items():
            prefix = Qwen21ComfyLoRA._block_prefix(Qwen21ComfyLoRA._gate_of(path))
            if prefix is None:
                name = Qwen21ComfyLoRA.CHECKPOINT_NAMES.get(path, path)
                down = weights[f"diffusion_model.{name}.lora_A.weight"]
                up = weights[f"diffusion_model.{name}.lora_B.weight"]
            else:
                gate = prefix + Qwen21ComfyLoRA.FUSED[0]
                down = weights[f"diffusion_model.{prefix}{Qwen21ComfyLoRA.GATE_UP}.lora_A.weight"]
                up = weights[f"diffusion_model.{prefix}{Qwen21ComfyLoRA.GATE_UP}.lora_B.weight"]
                rows, gate_rank = up.shape[0] // 2, ranks[gate]
                if path == gate:
                    down, up = down[:gate_rank], up[:rows, :gate_rank]
                else:
                    down, up = down[gate_rank:], up[rows:, gate_rank:]
            name = Qwen21ComfyLoRA.CHECKPOINT_NAMES.get(path, path)
            out[f"transformer.{name}.lora_A.weight"] = down
            out[f"transformer.{name}.lora_B.weight"] = up / scales[path]
            out[f"transformer.{name}.alpha"] = mx.array(scales[path] * rank)
        return out

    @staticmethod
    def _block_prefix(path: str) -> str | None:
        # "transformer_blocks.3" for "transformer_blocks.3.img_mlp.gate_layer", None for any other layer
        if path.endswith(Qwen21ComfyLoRA.FUSED[0]):
            return path[: -len(Qwen21ComfyLoRA.FUSED[0])]
        return None

    @staticmethod
    def _gate_of(path: str) -> str:
        if path.endswith(Qwen21ComfyLoRA.FUSED[1]):
            return path[: -len(Qwen21ComfyLoRA.FUSED[1])] + Qwen21ComfyLoRA.FUSED[0]
        return path
