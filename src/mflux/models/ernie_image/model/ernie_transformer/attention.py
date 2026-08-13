import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.ernie_image.model.ernie_transformer.rope_embedder import apply_rotary_emb


class ErnieAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float, qk_layernorm: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = [nn.Linear(hidden_size, hidden_size, bias=False)]

        if qk_layernorm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        B, S, _ = x.shape

        q = self.to_q(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.to_k(x).reshape(B, S, self.num_heads, self.head_dim)
        v = self.to_v(x).reshape(B, S, self.num_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        q, k = apply_rotary_emb(q, k, cos, sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.to_out[0](out)
