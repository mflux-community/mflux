import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear
from mflux.models.ideogram4.model.ideogram4_transformer.rope_embedder import Ideogram4MRoPE


class Qwen3VLAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        del max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = Fp8Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = Fp8Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = Fp8Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = Fp8Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        query_states = q_proj.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = k_proj.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = v_proj.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = Qwen3VLAttention.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.num_key_value_heads != self.num_attention_heads:
            key_states = Qwen3VLAttention.repeat_kv(key_states, self.num_key_value_groups)
            value_states = Qwen3VLAttention.repeat_kv(value_states, self.num_key_value_groups)

        attn_output = scaled_dot_product_attention(
            query_states.astype(mx.float32),
            key_states.astype(mx.float32),
            value_states.astype(mx.float32),
            scale=self.scaling,
            mask=attention_mask,
        )
        attn_output = attn_output.astype(hidden_states.dtype)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attn_output)

    @staticmethod
    def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def apply_rotary_pos_emb(
        q: mx.array,
        k: mx.array,
        cos: mx.array,
        sin: mx.array,
        unsqueeze_dim: int = 1,
    ) -> tuple[mx.array, mx.array]:
        cos = mx.expand_dims(cos, axis=unsqueeze_dim)
        sin = mx.expand_dims(sin, axis=unsqueeze_dim)
        q_embed = (q * cos) + (Ideogram4MRoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (Ideogram4MRoPE.rotate_half(k) * sin)
        return q_embed, k_embed
