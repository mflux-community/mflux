import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention


class Fibo2Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, kv_dim: int | None = None, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        kv_dim = kv_dim or dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(kv_dim, dim, bias=False)
        self.to_v = nn.Linear(kv_dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        freqs_cis: mx.array | None = None,
    ) -> mx.array:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch_size, seq_len, _ = hidden_states.shape
        kv_len = context.shape[1]

        query = self.norm_q(self.to_q(hidden_states).reshape(batch_size, seq_len, self.n_heads, self.head_dim))
        key = self.norm_k(self.to_k(context).reshape(batch_size, kv_len, self.n_heads, self.head_dim))
        value = self.to_v(context).reshape(batch_size, kv_len, self.n_heads, self.head_dim)

        if freqs_cis is not None:
            query = ZImageAttention.apply_rotary_emb(query, freqs_cis)
            key = ZImageAttention.apply_rotary_emb(key, freqs_cis)

        query, key, value = (mx.transpose(t, (0, 2, 1, 3)) for t in (query, key, value))
        out = scaled_dot_product_attention(query, key, value, scale=self.scale)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        return self.to_out[0](out)
