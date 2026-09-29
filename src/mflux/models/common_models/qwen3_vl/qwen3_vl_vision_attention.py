import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class Qwen3VLVisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        preserve_input_dtype: bool = False,
    ):
        super().__init__()
        self.dim = hidden_size
        self.preserve_input_dtype = preserve_input_dtype
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(1, 0, 2, 3)
        query_states, key_states, value_states = mx.split(qkv, 3, axis=0)
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = Qwen3VLVisionAttention._apply_rotary_pos_emb_vision(
                q=query_states,
                k=key_states,
                cos=cos,
                sin=sin,
            )
            if self.preserve_input_dtype:
                query_states = query_states.astype(value_states.dtype)
                key_states = key_states.astype(value_states.dtype)

        attn_outputs_chunks: list[mx.array] = []
        if cu_seqlens is not None and len(cu_seqlens) > 1:
            lengths = [int((cu_seqlens[i + 1] - cu_seqlens[i]).item()) for i in range(len(cu_seqlens) - 1)]
            offset = 0
            for length in lengths:
                q_chunk = query_states[offset : offset + length]
                k_chunk = key_states[offset : offset + length]
                v_chunk = value_states[offset : offset + length]
                offset += length

                q = q_chunk.transpose(1, 0, 2)
                k = k_chunk.transpose(1, 0, 2)
                v = v_chunk.transpose(1, 0, 2)
                q = mx.expand_dims(q, axis=0)
                k = mx.expand_dims(k, axis=0)
                v = mx.expand_dims(v, axis=0)

                attn_output = scaled_dot_product_attention(q, k, v, scale=self.scaling, mask=None)
                attn_output = attn_output.squeeze(0).transpose(1, 0, 2)
                attn_outputs_chunks.append(attn_output)

            attn_output = mx.concatenate(attn_outputs_chunks, axis=0)
        else:
            q = query_states.transpose(1, 0, 2)
            k = key_states.transpose(1, 0, 2)
            v = value_states.transpose(1, 0, 2)
            q = mx.expand_dims(q, axis=0)
            k = mx.expand_dims(k, axis=0)
            v = mx.expand_dims(v, axis=0)

            attn_output = scaled_dot_product_attention(q, k, v, scale=self.scaling, mask=None)
            attn_output = attn_output.squeeze(0).transpose(1, 0, 2)

        attn_output = attn_output.reshape(seq_length, self.dim)
        attn_output = self.proj(attn_output)
        return attn_output

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    @staticmethod
    def _apply_rotary_pos_emb_vision(
        q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
    ) -> tuple[mx.array, mx.array]:
        cos = cos[..., None, :]
        sin = sin[..., None, :]
        q_embed = (q * cos) + (Qwen3VLVisionAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3VLVisionAttention._rotate_half(k) * sin)
        return q_embed, k_embed
