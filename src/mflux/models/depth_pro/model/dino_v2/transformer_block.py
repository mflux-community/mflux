import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.dino_v2.attention import Attention
from mflux.models.depth_pro.model.dino_v2.layer_scale import LayerScale
from mflux.models.depth_pro.model.dino_v2.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 1024, num_heads: int = 16, mlp_hidden_dim: int = 4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims=dim, eps=1e-6, bias=True)
        self.attn = Attention(dim=dim, head_dim=dim // num_heads, num_heads=num_heads)
        self.ls1 = LayerScale(dims=dim, init_values=1e-5)
        self.norm2 = nn.LayerNorm(dims=dim, eps=1e-6, bias=True)
        self.mlp = MLP(dim=dim, hidden_dim=mlp_hidden_dim)
        self.ls2 = LayerScale(dims=dim, init_values=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
