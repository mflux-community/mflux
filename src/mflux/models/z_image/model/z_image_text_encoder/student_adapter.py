import mlx.core as mx
from mlx import nn


class ZImageStudentAdapter(nn.Module):
    def __init__(
        self,
        student_dim: int = 2048,
        teacher_dim: int = 2560,
        dim: int = 1024,
        heads: int = 8,
        blocks: int = 2,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.q_proj = nn.Linear(student_dim, dim)
        self.kv_proj = nn.Linear(student_dim, dim)
        self.blocks = [_CrossAttentionBlock(dim, heads, ff_mult) for _ in range(blocks)]
        self.proj_out = nn.Linear(dim, teacher_dim)

    def __call__(self, student_hidden: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        x = student_hidden.astype(mx.float32)
        if attention_mask is None:
            attention_mask = mx.ones(x.shape[:2], dtype=mx.bool_)
        valid = attention_mask.astype(mx.bool_)
        q = self.q_proj(x)
        # kv is computed once and shared by both blocks; only q is updated.
        kv = self.kv_proj(x)
        for block in self.blocks:
            q = block(q, kv, valid)
        return mx.where(valid[..., None], self.proj_out(q), 0.0)


class _CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array, valid: mx.array) -> mx.array:
        batch, q_len, dim = q.shape
        k_len = k.shape[1]
        head_dim = dim // self.heads
        q = self.query_proj(q).reshape(batch, q_len, self.heads, head_dim).transpose(0, 2, 1, 3)
        k = self.key_proj(k).reshape(batch, k_len, self.heads, head_dim).transpose(0, 2, 1, 3)
        v = self.value_proj(v).reshape(batch, k_len, self.heads, head_dim).transpose(0, 2, 1, 3)
        # Padded keys are masked out; a boolean mask is the same as torch's additive -inf mask.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask=valid[:, None, None, :])
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, q_len, dim))


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-5)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-5)
        self.attn = _CrossAttention(dim, heads)
        self.norm_ff = nn.LayerNorm(dim, eps=1e-5)
        self.ff_in = nn.Linear(dim, dim * ff_mult)
        self.ff_out = nn.Linear(dim * ff_mult, dim)

    def __call__(self, q: mx.array, kv: mx.array, valid: mx.array) -> mx.array:
        kv = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kv, kv, valid)
        # nn.GELU() in the torch module is the exact erf form, not the tanh approximation.
        return q + self.ff_out(nn.gelu(self.ff_in(self.norm_ff(q))))
