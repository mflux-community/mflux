import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_block import Qwen3VLVisionBlock
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_patch_embed import Qwen3VLVisionPatchEmbed
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_patch_merger import Qwen3VLVisionPatchMerger
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_rotary_embedding import Qwen3VLVisionRotaryEmbedding


class Qwen3VLVisionModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        depth: int = 24,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 2560,
        deepstack_visual_indexes: list[int] | None = None,
        hidden_act: str = "gelu_pytorch_tanh",
        preserve_input_dtype: bool = False,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.preserve_input_dtype = preserve_input_dtype
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings**0.5)

        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = [
            Qwen3VLVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                preserve_input_dtype=preserve_input_dtype,
            )
            for _ in range(depth)
        ]
        self.merger = Qwen3VLVisionPatchMerger(
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=out_hidden_size,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = deepstack_visual_indexes or [5, 11, 17]
        self.deepstack_merger_list = [
            Qwen3VLVisionPatchMerger(
                hidden_size=hidden_size,
                spatial_merge_size=spatial_merge_size,
                out_hidden_size=out_hidden_size,
                use_postshuffle_norm=True,
            )
            for _ in range(len(self.deepstack_visual_indexes))
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        return_deepstack: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = Qwen3VLVisionModel._fast_pos_embed_interpolate(self.spatial_merge_size, self.pos_embed, self.num_grid_per_side, grid_thw)  # fmt: off
        if self.preserve_input_dtype:
            pos_embeds = pos_embeds.astype(hidden_states.dtype)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = Qwen3VLVisionModel._rot_pos_emb(self.rotary_pos_emb, self.spatial_merge_size, grid_thw)

        # Prepare position embeddings for attention
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        position_embeddings = (cos, sin)

        # Compute cu_seqlens for variable-length sequences
        cu_seqlens_list = []
        for i in range(grid_thw.shape[0]):
            t, h, w = int(grid_thw[i, 0].item()), int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())
            seq_len_img = h * w * t
            cu_seqlens_list.extend([seq_len_img] * t)
        cu_seqlens = mx.array(
            [0] + [sum(cu_seqlens_list[: i + 1]) for i in range(len(cu_seqlens_list))], dtype=mx.int32
        )

        # Process through blocks
        deepstack_image_embeds = [] if return_deepstack else None
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

            if return_deepstack and layer_idx in self.deepstack_visual_indexes:
                deepstack_idx = self.deepstack_visual_indexes.index(layer_idx)
                deepstack_embeds = self.deepstack_merger_list[deepstack_idx](hidden_states)
                deepstack_image_embeds.append(deepstack_embeds)

        image_embeds = self.merger(hidden_states)
        return image_embeds, deepstack_image_embeds

    @staticmethod
    def _fast_pos_embed_interpolate(spatial_merge_size, pos_embed, num_grid_per_side, grid_thw: mx.array) -> mx.array:
        grid_ts = grid_thw[:, 0]
        grid_hs = grid_thw[:, 1]
        grid_ws = grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            h_idxs = mx.linspace(0, num_grid_per_side - 1, h)
            w_idxs = mx.linspace(0, num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.astype(mx.int32)
            w_idxs_floor = w_idxs.astype(mx.int32)
            h_idxs_ceil = mx.clip(h_idxs_floor + 1, 0, num_grid_per_side - 1)
            w_idxs_ceil = mx.clip(w_idxs_floor + 1, 0, num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor.astype(mx.float32)
            dw = w_idxs - w_idxs_floor.astype(mx.float32)

            base_h = h_idxs_floor * num_grid_per_side
            base_h_ceil = h_idxs_ceil * num_grid_per_side

            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
            ]

            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        # Convert to arrays
        max_len = max(len(idx_list[0]), len(idx_list[1]), len(idx_list[2]), len(idx_list[3]))
        idx_array = mx.zeros((4, max_len), dtype=mx.int32)
        weight_array = mx.zeros((4, max_len), dtype=mx.float32)

        for i in range(4):
            idx_array[i] = (
                mx.concatenate(
                    [
                        mx.array(idx_list[i][:max_len], dtype=mx.int32),
                        mx.zeros(max_len - len(idx_list[i]), dtype=mx.int32),
                    ]
                )
                if len(idx_list[i]) < max_len
                else mx.array(idx_list[i][:max_len], dtype=mx.int32)
            )
            weight_array[i] = (
                mx.concatenate(
                    [
                        mx.array(weight_list[i][:max_len], dtype=mx.float32),
                        mx.zeros(max_len - len(weight_list[i]), dtype=mx.float32),
                    ]
                )
                if len(weight_list[i]) < max_len
                else mx.array(weight_list[i][:max_len], dtype=mx.float32)
            )

        # Get position embeddings
        pos_embeds = pos_embed(idx_array) * weight_array[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Split by image
        patch_pos_embeds_list = []
        start = 0
        for h, w in zip(grid_hs, grid_ws):
            h, w = int(h.item()), int(w.item())
            end = start + h * w
            patch_pos_embeds_list.append(patch_pos_embeds[start:end])
            start = end

        # Permute and reshape for spatial merging
        patch_pos_embeds_permute = []
        merge_size = spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds_list, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            pos_embed = mx.tile(pos_embed, (t, 1))
            pos_embed = pos_embed.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            pos_embed = pos_embed.transpose(0, 1, 3, 2, 4, 5)
            pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])
            patch_pos_embeds_permute.append(pos_embed)

        patch_pos_embeds = mx.concatenate(patch_pos_embeds_permute)
        return patch_pos_embeds

    @staticmethod
    def _rot_pos_emb(rotary_pos_emb, spatial_merge_size, grid_thw: mx.array) -> mx.array:
        pos_ids = []
        for i in range(grid_thw.shape[0]):
            t, h, w = int(grid_thw[i, 0].item()), int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())

            hpos_ids = mx.repeat(mx.arange(h, dtype=mx.int32)[..., None], w, axis=1)
            wpos_ids = mx.repeat(mx.arange(w, dtype=mx.int32)[None, ...], h, axis=0)

            # Reshape for spatial merging
            merge_h = h // spatial_merge_size
            merge_w = w // spatial_merge_size
            hpos_ids = hpos_ids.reshape(merge_h, spatial_merge_size, merge_w, spatial_merge_size)
            wpos_ids = wpos_ids.reshape(merge_h, spatial_merge_size, merge_w, spatial_merge_size)

            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.reshape(-1)
            wpos_ids = wpos_ids.reshape(-1)

            pos_id_pair = mx.stack([hpos_ids, wpos_ids], axis=-1)
            if t > 1:
                pos_id_pair = mx.tile(pos_id_pair, (t, 1))

            pos_ids.append(pos_id_pair)

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = int(mx.max(grid_thw[:, 1:]).item())
        rotary_pos_emb_full = rotary_pos_emb(max_grid_size)

        h_indices = pos_ids[:, 0].astype(mx.int32)
        w_indices = pos_ids[:, 1].astype(mx.int32)
        h_emb = rotary_pos_emb_full[h_indices]
        w_emb = rotary_pos_emb_full[w_indices]
        rotary_pos_emb = mx.stack([h_emb, w_emb], axis=1)
        rotary_pos_emb = rotary_pos_emb.reshape(rotary_pos_emb.shape[0], -1)
        return rotary_pos_emb
