import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_5.qwen3_5_mrope import Qwen35MRoPE
from mflux.models.common_models.qwen3_5.qwen3_5_rms_norm import Qwen35RMSNorm


class Qwen35Attention(nn.Module):
    # Causal attention with an output gate and partial rotary embedding. For text input the
    # three M-RoPE position streams are equal, so the rotation is a standard 1D RoPE. A caller
    # with image tokens passes the (cos, sin) of Qwen35MRoPE instead.
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rotary_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
        fused_norm: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(head_dim, eps=rms_norm_eps, fused=fused_norm)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=rms_norm_eps, fused=fused_norm)

    def __call__(self, hidden_states: mx.array, rotary: tuple[mx.array, mx.array] | None = None) -> mx.array:
        batch, seq_len, _ = hidden_states.shape

        query_and_gate = self.q_proj(hidden_states).reshape(batch, seq_len, self.num_attention_heads, 2 * self.head_dim)
        query, gate = mx.split(query_and_gate, 2, axis=-1)
        gate = gate.reshape(batch, seq_len, -1)

        query = self.q_norm(query).transpose(0, 2, 1, 3)
        key = self.k_norm(self.k_proj(hidden_states).reshape(batch, seq_len, self.num_key_value_heads, self.head_dim))
        key = key.transpose(0, 2, 1, 3)
        value = self.v_proj(hidden_states).reshape(batch, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.transpose(0, 2, 1, 3)

        if rotary is None:
            query = self._rotate(query)
            key = self._rotate(key)
        else:
            cos, sin = rotary[0][:, None], rotary[1][:, None]
            query = Qwen35MRoPE.apply(query, cos, sin)
            key = Qwen35MRoPE.apply(key, cos, sin)

        attn = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale, mask="causal")
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(attn * mx.sigmoid(gate))

    def _rotate(self, x: mx.array) -> mx.array:
        return mx.fast.rope(x, self.rotary_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0)
