import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common.compute_precision import ComputePrecision


class ZImageAttention(nn.Module):
    # In float16 the output reaches ~1.4e5, so V is taken this many times smaller and the output scaled back
    # in the stream dtype. Attention is linear in V and there are no biases: exact up to rounding.
    FLOAT16_HEADROOM = 4.0

    def __init__(
        self,
        dim: int,
        n_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        # Projections (no bias)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]

        # Optional QK normalization
        if qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None
        self.compute_precision = ComputePrecision()

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        freqs_cis: mx.array | None = None,
    ) -> mx.array:
        precision = self.compute_precision
        dtype = hidden_states.dtype
        hidden_states = precision.to_compute(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = precision.shrink(self.to_v(hidden_states), ZImageAttention.FLOAT16_HEADROOM)

        # Reshape to (batch, seq_len, heads, head_dim)
        query = query.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply QK normalization
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Apply RoPE
        if freqs_cis is not None:
            query = precision.to_compute(ZImageAttention._apply_rotary_emb(query, freqs_cis))
            key = precision.to_compute(ZImageAttention._apply_rotary_emb(key, freqs_cis))

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        query = mx.transpose(query, axes=(0, 2, 1, 3))
        key = mx.transpose(key, axes=(0, 2, 1, 3))
        value = mx.transpose(value, axes=(0, 2, 1, 3))

        # Convert boolean mask to additive mask for SDPA
        mask = None
        if attention_mask is not None:
            mask = mx.where(
                attention_mask[:, None, None, :],
                mx.array(0.0, dtype=query.dtype),
                mx.array(float("-inf"), dtype=query.dtype),
            )

        hidden_states = scaled_dot_product_attention(query, key, value, scale=self.scale, mask=mask)
        hidden_states = mx.transpose(hidden_states, axes=(0, 2, 1, 3))
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.dim)
        hidden_states = self.to_out[0](hidden_states)
        return precision.to_stream(hidden_states, dtype, ZImageAttention.FLOAT16_HEADROOM)

    @staticmethod
    def _apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
        batch_size, seq_len, n_heads, head_dim = x.shape
        x = x.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
        freqs_cis = mx.expand_dims(freqs_cis, axis=0)
        freqs_cis = mx.expand_dims(freqs_cis, axis=2)
        x_real, x_imag = x[..., 0], x[..., 1]
        freqs_cos, freqs_sin = freqs_cis[..., 0], freqs_cis[..., 1]
        x_out_real = x_real * freqs_cos - x_imag * freqs_sin
        x_out_imag = x_real * freqs_sin + x_imag * freqs_cos
        x_out = mx.stack([x_out_real, x_out_imag], axis=-1)
        return x_out.reshape(batch_size, seq_len, n_heads, head_dim)
