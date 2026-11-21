import mlx.core as mx
from mlx import nn


class DreamLiteRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, hidden: mx.array) -> mx.array:
        value = hidden.astype(mx.float32)
        value = value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
        return value.astype(hidden.dtype) * self.weight


class DreamLiteAttention(nn.Module):
    # Multi-query attention: every query head reads the same single key/value head
    HEAD_DIM = 64

    def __init__(self, dim: int, num_heads: int, context_dim: int | None = None):
        super().__init__()
        head_dim = DreamLiteAttention.HEAD_DIM
        self.num_heads = num_heads
        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(context_dim or dim, head_dim, bias=False)
        self.to_v = nn.Linear(context_dim or dim, head_dim, bias=False)
        self.norm_q = DreamLiteRMSNorm(head_dim)
        self.norm_k = DreamLiteRMSNorm(head_dim)
        self.to_out = [nn.Linear(num_heads * head_dim, dim)]

    def __call__(self, hidden: mx.array, context: mx.array | None = None) -> mx.array:
        context = hidden if context is None else context
        batch, tokens, _ = hidden.shape
        head_dim = DreamLiteAttention.HEAD_DIM
        query = self.to_q(hidden).reshape(batch, tokens, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        # One key/value head: every query head attends to the same keys and values
        shared = (batch, self.num_heads, context.shape[1], head_dim)
        key = mx.broadcast_to(self.norm_k(self.to_k(context))[:, None], shared)
        value = mx.broadcast_to(self.to_v(context)[:, None], shared)
        attended = mx.fast.scaled_dot_product_attention(self.norm_q(query), key, value, scale=head_dim**-0.5)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, self.num_heads * head_dim)
        return self.to_out[0](attended)
