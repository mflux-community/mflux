import mlx.core as mx
from mlx import nn

from mflux.models.common_models.aiartlab_adapter import AdapterAttentionMixer, AdapterPerLayerNorm


class Qwen3SmallTextAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int = 6144,
        out_dim: int = 7680,
        hidden: int = 8192,
        n_slices: int = 6,
        attn_dim: int = 1024,
        attn_heads: int = 8,
        attn_blocks: int = 2,
        max_len: int = 256,
    ):
        super().__init__()
        self.max_len = max_len
        self.mlp = [
            AdapterPerLayerNorm(in_dim, n_slices),
            nn.Linear(in_dim, hidden),
            nn.GELU(approx="tanh"),
            nn.Linear(hidden, hidden),
            nn.GELU(approx="tanh"),
            nn.Linear(hidden, out_dim),
        ]
        self.attn = AdapterAttentionMixer(in_dim, out_dim, attn_dim, attn_heads, attn_blocks, max_len)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        return self.apply_mlp(x) + self.attn(x)

    def apply_mlp(self, x: mx.array) -> mx.array:
        for layer in self.mlp:
            x = layer(x)
        return x
