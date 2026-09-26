import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder_layer import Qwen3VLDecoderLayer
from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.common_models.qwen3_vl.qwen3_vl_rope import Qwen3VLRotaryEmbedding
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel


class Qwen21TextEncoder(nn.Module):
    # Qwen3-VL text stack of Qwen-Image-2.1: interleaved mrope, but text-only inputs use
    # one shared position ladder replicated across the three mrope axes.

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 12288,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        mrope_section: list[int] | None = None,
        with_visual: bool = False,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                mrope_section=mrope_section,
                attention_bias=False,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        # the checkpoint ships an UNTIED language-model head; the diffusion path never
        # touches it, but grounded decoding (locate_object, edit variant only) needs it
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False) if with_visual else None
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            mrope_section=mrope_section,
        )
        # Vision tower of the Qwen3-VL encoder. Only instantiated for the edit variant;
        # t2i never touches it so it stays None and maps no weights.
        self.visual = (
            Qwen3VLVisionModel(
                patch_size=16,
                temporal_patch_size=2,
                in_channels=3,
                hidden_size=1152,
                num_heads=16,
                intermediate_size=4304,
                depth=27,
                spatial_merge_size=2,
                num_position_embeddings=2304,
                out_hidden_size=hidden_size,
                deepstack_visual_indexes=[8, 16, 24],
                hidden_act="gelu_pytorch_tanh",
            )
            if with_visual
            else None
        )

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        mask_dtype = mx.float32
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = mx.where(
            idx[None, :] > idx[:, None],
            mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
            mx.zeros((seq_len, seq_len), dtype=mask_dtype),
        )
        attention_mask_4d = mx.broadcast_to(causal[None, None, :, :], (batch_size, 1, seq_len, seq_len)) + padding_mask

        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask_4d, position_embeddings)

        return self.norm(hidden_states)

    @staticmethod
    def build_mrope_positions(
        input_ids: mx.array,
        image_mask: mx.array,
        image_grid_thw: mx.array,
        spatial_merge_size: int = 2,
    ) -> mx.array:
        # Port of transformers Qwen3VL.get_rope_index for a single unpadded sequence:
        # text runs share a 1D ladder on all three mrope axes; image tokens get 2D
        # (height, width) positions in merged-grid space and advance the ladder by
        # max(grid_h, grid_w), matching the reference position bookkeeping.
        mask = np.array(image_mask[0], dtype=bool)
        seq_len = mask.shape[0]
        positions = np.zeros((3, seq_len), dtype=np.int32)
        grids = [tuple(int(v) for v in grid) for grid in image_grid_thw.tolist()] if image_grid_thw is not None else []
        grid_index = 0
        pos = 0
        i = 0
        while i < seq_len:
            if mask[i]:
                _, grid_h, grid_w = grids[grid_index]
                grid_index += 1
                llm_h, llm_w = grid_h // spatial_merge_size, grid_w // spatial_merge_size
                n = llm_h * llm_w
                t_axis = np.full(1, pos, dtype=np.int32)  # single frame: temporal position = pos
                h_axis = np.arange(llm_h, dtype=np.int32) + pos
                w_axis = np.arange(llm_w, dtype=np.int32) + pos
                t_grid, h_grid, w_grid = np.meshgrid(t_axis, h_axis, w_axis, indexing="ij")
                positions[:, i : i + n] = np.stack([t_grid.reshape(-1), h_grid.reshape(-1), w_grid.reshape(-1)])
                pos += max(llm_h, llm_w)
                i += n
            else:
                positions[:, i] = pos
                pos += 1
                i += 1
        return mx.array(positions)

    def _embed_with_vision(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None,
        image_grid_thw: mx.array | None,
        image_token_id: int = 151655,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list, mx.array]:
        # Shared prologue for the edit-mode forward and grounded decoding: embeddings
        # with vision features scattered into <|image_pad|> positions, mrope positions,
        # and the deepstack features kept separate for the layer loop to inject.
        # pixel_values=None is a text-only sequence (grounding probes, no image).
        if pixel_values is not None and self.visual is None:
            raise RuntimeError("vision features require the vision tower (Qwen21TextEncoder(with_visual=True))")

        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if pixel_values is None:
            text_mask = mx.zeros(input_ids.shape, dtype=mx.bool_)
            positions = Qwen21TextEncoder.build_mrope_positions(input_ids, text_mask, None)
            keep = mx.zeros(input_ids.shape + (1,), dtype=mx.bool_)
            return (
                hidden_states,
                text_mask,
                mx.zeros(input_ids.shape, dtype=mx.int32),
                keep,
                [],
                positions,
            )

        image_embeds, deepstack_embeds = self.visual(pixel_values, image_grid_thw, return_deepstack=True)
        image_mask = input_ids == image_token_id  # (1, seq_len)
        n_image_tokens = int(mx.sum(image_mask).item())
        if n_image_tokens != image_embeds.shape[0]:
            raise ValueError(
                f"<|image_pad|> count {n_image_tokens} != vision embeds {image_embeds.shape[0]} -- template/processor mismatch"
            )

        # Scatter the vision embeds into the embedding sequence in order.
        mask_flat = image_mask[0]
        gather_index = mx.cumsum(mask_flat.astype(mx.int32)) - 1  # k-th True -> embed k
        gathered = image_embeds[gather_index].astype(hidden_states.dtype)
        keep = mask_flat[:, None]  # (seq_len, 1)
        hidden_states = mx.where(keep[None, :, :], gathered[None, :, :], hidden_states)
        positions = Qwen21TextEncoder.build_mrope_positions(input_ids, image_mask, image_grid_thw)
        return hidden_states, image_mask, gather_index, keep, deepstack_embeds, positions

    def forward_vl(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        image_token_id: int = 151655,
    ) -> tuple[mx.array, mx.array]:
        # Edit-mode forward: vision embeds replace <|image_pad|> positions, deepstack
        # features inject at image positions after the first three decoder layers, and
        # the hidden states are returned BEFORE the final RMSNorm (what the diffusion
        # transformer was trained on). Also returns the (1, seq_len) <|image_pad|> mask.
        batch_size, seq_len = input_ids.shape
        hidden_states, image_mask, gather_index, keep, deepstack_embeds, positions = self._embed_with_vision(
            input_ids, pixel_values, image_grid_thw, image_token_id
        )
        position_embeddings = self.rotary_emb(hidden_states, positions[:, None, :])  # (3, batch=1, seq)

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = mx.where(
            idx[None, :] > idx[:, None],
            mx.full((seq_len, seq_len), -float("inf"), dtype=mx.float32),
            mx.zeros((seq_len, seq_len), dtype=mx.float32),
        )
        attention_mask_4d = causal[None, None, :, :]

        for layer_index, layer in enumerate(self.layers):
            hidden_states, _ = layer(hidden_states, attention_mask_4d, position_embeddings)
            if layer_index < len(deepstack_embeds):
                # inject at image positions only: expand the (n_image, hidden) deepstack
                # features to full sequence length via the same gather index
                ds_full = deepstack_embeds[layer_index][gather_index].astype(hidden_states.dtype)
                hidden_states = mx.where(keep[None, :, :], (hidden_states[0] + ds_full)[None, :, :], hidden_states)

        return hidden_states, image_mask

    def locate_object(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
        image_token_id: int = 151655,
        max_new_tokens: int = 64,
        stop_token_ids: tuple[int, ...] = (151645, 151643),
    ) -> list[int]:
        # Grounded decoding with the same language model the edit prompt encoder uses.
        return self.generate(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )

    def generate(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        image_token_id: int = 151655,
        max_new_tokens: int = 64,
        stop_token_ids: tuple[int, ...] = (151645, 151643),
    ) -> list[int]:
        # Greedy decoding with the same language model the edit prompt encoder uses:
        # one vision+prompt prefill into a per-layer KV cache, then greedy single-token
        # steps. The encoder is a full Qwen3-VL with its own untied lm_head, so this
        # runs without any extra weights; deepstack injects during the prefill exactly
        # as in forward_vl, and text-only continuation tokens need none. Serves
        # grounding (locate_object), prompt rewriting, and output verification.
        if self.lm_head is None:
            raise RuntimeError("generate() requires the language-model head (Qwen21TextEncoder(with_visual=True))")
        hidden_states, _, gather_index, keep, deepstack_embeds, positions = self._embed_with_vision(
            input_ids, pixel_values, image_grid_thw, image_token_id
        )
        seq_len = hidden_states.shape[1]
        position_embeddings = self.rotary_emb(hidden_states, positions[:, None, :])

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = mx.where(
            idx[None, :] > idx[:, None],
            mx.full((seq_len, seq_len), -float("inf"), dtype=mx.float32),
            mx.zeros((seq_len, seq_len), dtype=mx.float32),
        )
        total_length = seq_len + max_new_tokens
        caches = []
        for layer_index, layer in enumerate(self.layers):
            hidden_states, cache = layer(
                hidden_states, causal[None, None, :, :], position_embeddings, max_cache_length=total_length
            )
            if layer_index < len(deepstack_embeds):
                ds_full = deepstack_embeds[layer_index][gather_index].astype(hidden_states.dtype)
                hidden_states = mx.where(keep[None, :, :], (hidden_states[0] + ds_full)[None, :, :], hidden_states)
            caches.append(cache)
        mx.eval(hidden_states)

        last_position = int(positions[0, -1])
        generated: list[int] = []
        for step in range(max_new_tokens):
            logits = self.lm_head(self.norm(hidden_states[:, -1]))
            next_id = int(mx.argmax(logits[0]).item())
            if next_id in stop_token_ids:
                break
            generated.append(next_id)

            token_hidden = self.embed_tokens(mx.array([[next_id]])).astype(hidden_states.dtype)
            token_position = mx.full((3, 1, 1), last_position + 1 + step, dtype=mx.int32)
            token_embeddings = self.rotary_emb(token_hidden, token_position)
            for layer_index, layer in enumerate(self.layers):
                token_hidden, cache = layer(token_hidden, None, token_embeddings, past_key_value=caches[layer_index])
                caches[layer_index] = cache
            hidden_states = token_hidden
            mx.eval(hidden_states)
        return generated
