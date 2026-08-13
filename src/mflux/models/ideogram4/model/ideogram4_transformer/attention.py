import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear
from mflux.models.ideogram4.model.ideogram4_transformer.modulation import Ideogram4RMSNorm
from mflux.models.ideogram4.model.ideogram4_transformer.rope_embedder import Ideogram4MRoPE


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.qkv = Fp8Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = Fp8Linear(hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q = self.norm_q(q).transpose(0, 2, 1, 3)
        k = self.norm_k(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q, k = Ideogram4MRoPE.apply_rotary_pos_emb(q, k, cos, sin)
        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        mask = mx.where(
            same_segment[:, None, :, :],
            mx.zeros((batch_size, 1, seq_len, seq_len), dtype=mx.float32),
            mx.full((batch_size, 1, seq_len, seq_len), -float("inf"), dtype=mx.float32),
        )
        out = scaled_dot_product_attention(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scaling,
            mask=mask,
        )
        out = out.astype(x.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)
