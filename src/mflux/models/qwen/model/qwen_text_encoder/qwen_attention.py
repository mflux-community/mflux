import math

import mlx.core as mx
from mlx import nn

from .qwen_rope import QwenRotaryEmbedding


class QwenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int = None,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
        rope_scaling: dict = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // self.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )
        self.rope_scaling = rope_scaling or {"mrope_section": [16, 24, 24]}

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        query_states, key_states = QwenAttention._apply_multimodal_rotary_pos_emb(
            q=query_states,
            k=key_states,
            position_embeddings=position_embeddings,
            mrope_section=self.rope_scaling["mrope_section"],
        )

        if self.num_key_value_heads != self.num_attention_heads:
            key_states = QwenAttention._repeat_kv(key_states, self.num_key_value_groups)
            value_states = QwenAttention._repeat_kv(value_states, self.num_key_value_groups)

        mask = (
            attention_mask[:, :, :, : key_states.shape[-2]].astype(query_states.dtype)
            if attention_mask is not None
            else None
        )
        attn_output = mx.fast.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            mask=mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    @staticmethod
    def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def _apply_multimodal_rotary_pos_emb(
        q: mx.array,
        k: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        mrope_section: list[int],
        unsqueeze_dim: int = 1,
    ) -> tuple[mx.array, mx.array]:
        mrope_section_doubled = [s * 2 for s in mrope_section]

        cos, sin = position_embeddings
        cos_chunks = []
        sin_chunks = []
        start_idx = 0
        for section_size in mrope_section_doubled:
            end_idx = start_idx + section_size
            cos_chunk = cos[..., start_idx:end_idx]
            sin_chunk = sin[..., start_idx:end_idx]
            cos_chunks.append(cos_chunk)
            sin_chunks.append(sin_chunk)
            start_idx = end_idx

        cos_selected = [chunk[i % 3] for i, chunk in enumerate(cos_chunks)]
        sin_selected = [chunk[i % 3] for i, chunk in enumerate(sin_chunks)]

        cos_combined = mx.concatenate(cos_selected, axis=-1)
        sin_combined = mx.concatenate(sin_selected, axis=-1)

        if unsqueeze_dim == 1:
            cos_combined = mx.expand_dims(cos_combined, axis=1)
            sin_combined = mx.expand_dims(sin_combined, axis=1)

        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        cos_combined = cos_combined.astype(mx.float32)
        sin_combined = sin_combined.astype(mx.float32)

        q_embed = (q * cos_combined) + (QwenAttention._rotate_half(q) * sin_combined)
        k_embed = (k * cos_combined) + (QwenAttention._rotate_half(k) * sin_combined)

        q_embed = q_embed.astype(orig_q_dtype)
        k_embed = k_embed.astype(orig_k_dtype)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)
