import mlx.core as mx
from mlx import nn

from mflux.models.dreamlite.model.dreamlite_unet.attention import DreamLiteAttention


class DreamLiteGEGLU(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2)

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden, gate = mx.split(self.proj(hidden), 2, axis=-1)
        return hidden * nn.gelu(gate.astype(mx.float32)).astype(hidden.dtype)


class DreamLiteFeedForward(nn.Module):
    MULT = 3

    def __init__(self, dim: int):
        super().__init__()
        inner_dim = dim * DreamLiteFeedForward.MULT
        # Index 1 is the reference's dropout: it holds no weights and keeps the checkpoint's net.2 name
        self.net = [DreamLiteGEGLU(dim, inner_dim), nn.Identity(), nn.Linear(inner_dim, dim)]

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.net[2](self.net[0](hidden))


class DreamLiteTransformerBlock(nn.Module):
    EPS = 1e-5

    def __init__(self, dim: int, num_heads: int, context_dim: int, self_attention: bool):
        super().__init__()
        eps = DreamLiteTransformerBlock.EPS
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        if self_attention:
            self.attn1 = DreamLiteAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.attn2 = DreamLiteAttention(dim, num_heads, context_dim=context_dim)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ff = DreamLiteFeedForward(dim)

    def __call__(self, hidden: mx.array, context: mx.array) -> mx.array:
        normed = self.norm1(hidden)
        # A block without self-attention continues from the normalized states, with no residual, as the reference does
        hidden = hidden + self.attn1(normed) if "attn1" in self else normed
        hidden = hidden + self.attn2(self.norm2(hidden), context)
        return hidden + self.ff(self.norm3(hidden))
