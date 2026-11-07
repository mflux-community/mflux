import mlx.core as mx
from mlx import nn


class AdapterAttentionBlock(nn.Module):
    # Pre-norm transformer block of the AiArtLab text adapters (torch nn.MultiheadAttention + GELU FFN).
    def __init__(self, dim: int, heads: int, ffn_mult: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiHeadAttention(dim, heads, bias=True)
        self.n2 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = [nn.Linear(dim, dim * ffn_mult), nn.GELU(approx="tanh"), nn.Linear(dim * ffn_mult, dim)]

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        h = self.n1(x)
        x = x + self.attn(h, h, h, mask=attention_mask)
        h = self.n2(x)
        for layer in self.ffn:
            h = layer(h)
        return x + h
