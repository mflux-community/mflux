import mlx.core as mx
from mlx import nn

from mflux.models.anima.model.anima_transformer.anima_rope import AnimaRoPE
from mflux.models.common.compute_precision import ComputePrecision


class AnimaAttention(nn.Module):
    # Self-attention when the context is the input itself, cross-attention otherwise. Only self-attention
    # gets the rotary embedding; the query and key are RMS-normalized per head first.
    def __init__(self, dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(context_dim, dim, bias=False)
        self.v_proj = nn.Linear(context_dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.compute_precision = ComputePrecision()

    def __call__(self, x: mx.array, context: mx.array, rope: tuple[mx.array, mx.array] | None = None) -> mx.array:
        precision = self.compute_precision
        dtype = x.dtype
        x, context = precision.to_compute(x), precision.to_compute(context)
        batch, seq_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, seq_len, self.num_heads, -1))
        k = self.k_norm(self.k_proj(context).reshape(batch, context.shape[1], self.num_heads, -1))
        v = self.v_proj(context).reshape(batch, context.shape[1], self.num_heads, -1)
        if rope is not None:
            q = AnimaRoPE.rotate(q, *rope)
            k = AnimaRoPE.rotate(k, *rope)
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            scale=q.shape[-1] ** -0.5,
        )
        out = self.output_proj(out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))
        return precision.to_stream(out, dtype)
