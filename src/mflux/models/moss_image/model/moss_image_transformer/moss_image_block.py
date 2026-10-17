import math

import mlx.core as mx
import mlx.nn as nn

from mflux.models.moss_image.model.moss_image_transformer.moss_image_rope import rotate_half


def situ(hidden: mx.array, gate: mx.array) -> mx.array:
    return (4.0 * mx.tanh(gate / 4.0) * mx.sigmoid(gate)) * (25.0 * mx.tanh(hidden / 25.0))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return (x * norm) * self.weight


class MOSSImageBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Linear layers (proj.q, k, v, o, ffn_up_gate, ffn_down)
        self.proj = {}
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_o = nn.Linear(dim, dim, bias=False)
        self.proj_ffn_up_gate = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj_ffn_down = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x: mx.array, rope: mx.array, modulation: mx.array) -> mx.array:
        # modulation: (B, 6, dim)
        shift_a = modulation[:, 0:1, :]
        scale_a = modulation[:, 1:2, :]
        gate_a  = modulation[:, 2:3, :]
        shift_f = modulation[:, 3:4, :]
        scale_f = modulation[:, 4:5, :]
        gate_f  = modulation[:, 5:6, :]

        y = self.norm1(x) * (1.0 + scale_a) + shift_a

        q = self.proj_q(y)
        k = self.proj_k(y)
        v = self.proj_v(y)

        B, L, _ = q.shape
        q = mx.reshape(q, (B, L, self.heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.heads, self.head_dim))
        v = mx.reshape(v, (B, L, self.heads, self.head_dim))

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rotate_half(q, rope)
        k = rotate_half(k, rope)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = mx.reshape(mx.transpose(attn, (0, 2, 1, 3)), (B, L, self.dim))

        x = x + self.proj_o(attn) * gate_a

        y2 = self.norm2(x) * (1.0 + scale_f) + shift_f
        up_gate = self.proj_ffn_up_gate(y2)
        hidden, gate = mx.split(up_gate, 2, axis=-1)

        x = x + self.proj_ffn_down(situ(hidden, gate)) * gate_f
        return x


class MOSSImageTextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=True)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=True)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=True)

    def __call__(self, x: mx.array, rope: mx.array) -> mx.array:
        qkv = self.qkv(self.norm1(x))
        q, k, v = mx.split(qkv, 3, axis=-1)

        B, L, _ = q.shape
        q = mx.reshape(q, (B, L, self.heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.heads, self.head_dim))
        v = mx.reshape(v, (B, L, self.heads, self.head_dim))

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rotate_half(q, rope)
        k = rotate_half(k, rope)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = mx.reshape(mx.transpose(attn, (0, 2, 1, 3)), (B, L, self.dim))

        x = x + self.proj(attn)
        y = self.norm2(x)
        x = x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))
        return x
