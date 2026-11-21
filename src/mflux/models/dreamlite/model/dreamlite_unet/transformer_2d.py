import mlx.core as mx
from mlx import nn

from mflux.models.dreamlite.model.dreamlite_unet.transformer_block import DreamLiteTransformerBlock


class DreamLiteTransformer2D(nn.Module):
    GROUPS = 32
    EPS = 1e-6

    def __init__(self, dim: int, num_heads: int, num_layers: int, context_dim: int, self_attention: bool):
        super().__init__()
        self.norm = nn.GroupNorm(
            DreamLiteTransformer2D.GROUPS, dim, eps=DreamLiteTransformer2D.EPS, pytorch_compatible=True
        )
        self.proj_in = nn.Linear(dim, dim)
        self.transformer_blocks = [
            DreamLiteTransformerBlock(dim, num_heads, context_dim, self_attention) for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(dim, dim)

    def __call__(self, hidden: mx.array, context: mx.array) -> mx.array:
        batch, height, width, channels = hidden.shape
        residual = hidden
        hidden = self.proj_in(self.norm(hidden).reshape(batch, height * width, channels))
        for block in self.transformer_blocks:
            hidden = block(hidden, context)
        return self.proj_out(hidden).reshape(batch, height, width, channels) + residual
