import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.krea2.model.krea2_transformer.common import Krea2QKNorm
from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder


class Krea2Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int | None = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(dim, self.head_dim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.head_dim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.head_dim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = Krea2QKNorm(self.head_dim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def __call__(self, x: mx.array, freqs: mx.array | None = None, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape
        q = self.wq(x).reshape(B, L, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.kvheads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.kvheads, self.head_dim).transpose(0, 2, 1, 3)
        gate = self.gate(x)

        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = Krea2RopeEmbedder.apply_rope(q, k, freqs)

        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)

        out = scaled_dot_product_attention(q.astype(v.dtype), k.astype(v.dtype), v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(out * mx.sigmoid(gate))
