import mlx.core as mx

from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping


class Fibo2ComfyLoRA:
    # Writes trained FIBO 2 LoRAs in ComfyUI's layout (Comfy-Org/Fibo-2.0), the one ComfyUI and ai-toolkit read:
    # diffusion_model.* keys, q/k/v fused into attention.qkv, cross-attention k/v into to_kv, the feed-forward gate and
    # up into w13, and the feed-forward zero-padded from 9536 to 9728 as in that repack. Parts of a fused layer are
    # stacked exactly: their down weights one above the other and their up weights block-diagonal, so each part keeps
    # its own delta and the fused rank is the sum. Each part's alpha / rank is folded into its up weight.
    PADDED_FFN = 9728
    # Metadata keys with each trained module's rank and scale (alpha / rank), so a resumed run splits the fused layers
    # back exactly and each module trains on at its own scale
    RANKS_KEY = "mflux_lora_ranks"
    SCALES_KEY = "mflux_lora_scales"

    @staticmethod
    def weights(loras: dict[str, tuple[mx.array, mx.array, float]]) -> dict[str, mx.array]:
        # loras: MLX module path -> (down (rank, in), up (out, rank), scale)
        groups: dict[str, list[tuple[int, int, mx.array, mx.array, int]]] = {}
        for path, (down, up, scale) in loras.items():
            name, index, count = Fibo2ComfyLoRA._comfy_name(path)
            groups.setdefault(name, []).append((index, count, down, up * scale, up.shape[0]))
        out = {}
        for name, parts in groups.items():
            down, up = Fibo2ComfyLoRA._stack(name, parts)
            out[f"diffusion_model.{name}.lora_A.weight"] = down
            out[f"diffusion_model.{name}.lora_B.weight"] = up
        return out

    @staticmethod
    def _comfy_name(path: str) -> tuple[str, int, int]:
        # (ComfyUI module name, index of this part in the fused layer, number of parts)
        fused = {
            ".attention.to_q": (".attention.qkv", 0, 3),
            ".attention.to_k": (".attention.qkv", 1, 3),
            ".attention.to_v": (".attention.qkv", 2, 3),
            "_attention.to_k": ("_attention.to_kv", 0, 2),
            "_attention.to_v": ("_attention.to_kv", 1, 2),
            ".feed_forward.w1": (".feed_forward.w13", 0, 2),
            ".feed_forward.w3": (".feed_forward.w13", 1, 2),
            ".ffn.w1": (".ffn.w13", 0, 2),
            ".ffn.w3": (".ffn.w13", 1, 2),
        }
        for suffix, (comfy, index, count) in fused.items():
            if path.endswith(suffix):
                return path[: -len(suffix)] + comfy, index, count
        if path.endswith(".attention.to_out.0"):
            return path[: -len(".attention.to_out.0")] + ".attention.out", 0, 1
        if path.endswith("_attention.to_out.0"):
            return path[: -len(".to_out.0")] + ".to_out", 0, 1
        return Fibo2LoRAMapping.diffusers_name(path), 0, 1

    @staticmethod
    def _stack(name: str, parts: list[tuple[int, int, mx.array, mx.array, int]]) -> tuple[mx.array, mx.array]:
        count = parts[0][1]
        ffn = name.endswith(("feed_forward.w13", "ffn.w13"))
        rows = parts[0][4]
        block = Fibo2ComfyLoRA.PADDED_FFN if ffn and rows == Fibo2LoRAMapping.FFN_HIDDEN else rows
        downs = [down for _, _, down, _, _ in sorted(parts, key=lambda p: p[0])]
        rank = sum(d.shape[0] for d in downs)
        up = mx.zeros((block * count, rank), dtype=parts[0][3].dtype)
        column = 0
        for index, _, down, part_up, out in sorted(parts, key=lambda p: p[0]):
            up[index * block : index * block + out, column : column + down.shape[0]] = part_up
            column += down.shape[0]
        down = mx.concatenate(downs, axis=0)
        if name.endswith(("feed_forward.w2", "ffn.w2")) and down.shape[1] == Fibo2LoRAMapping.FFN_HIDDEN:
            down = mx.pad(down, [(0, 0), (0, Fibo2ComfyLoRA.PADDED_FFN - down.shape[1])])
        return down, up

    @staticmethod
    def split(
        weights: dict[str, mx.array], ranks: dict[str, int], scales: dict[str, float] | None = None
    ) -> dict[str, mx.array]:
        # The inverse of weights() for a run's own checkpoint: each module gets back its own down and up weights at its
        # own rank, under the names Fibo2LoRAMapping reads as unfused layers. With the scales, alpha / rank comes out
        # of up again and is written as .alpha; without them (an older checkpoint) it stays folded into up
        groups: dict[str, list[tuple[int, int, str, int]]] = {}
        for path, rank in ranks.items():
            name, index, count = Fibo2ComfyLoRA._comfy_name(path)
            groups.setdefault(name, []).append((index, count, path, rank))
        out = {}
        for name, parts in groups.items():
            down = weights[f"diffusion_model.{name}.lora_A.weight"]
            up = weights[f"diffusion_model.{name}.lora_B.weight"]
            block = up.shape[0] // parts[0][1]
            ffn = name.endswith(("feed_forward.w13", "ffn.w13"))
            rows = Fibo2LoRAMapping.FFN_HIDDEN if ffn and block == Fibo2ComfyLoRA.PADDED_FFN else block
            column = 0
            for index, _, path, rank in sorted(parts):
                part_down = down[column : column + rank]
                if name.endswith(("feed_forward.w2", "ffn.w2")) and part_down.shape[1] == Fibo2ComfyLoRA.PADDED_FFN:
                    part_down = part_down[:, : Fibo2LoRAMapping.FFN_HIDDEN]
                source = f"transformer.{Fibo2LoRAMapping.diffusers_name(path)}"
                part_up = up[index * block : index * block + rows, column : column + rank]
                out[f"{source}.lora_A.weight"] = part_down
                if scales is not None and path in scales:
                    part_up = part_up / scales[path]
                    out[f"{source}.alpha"] = mx.array(scales[path] * rank)
                out[f"{source}.lora_B.weight"] = part_up
                column += rank
        return out
