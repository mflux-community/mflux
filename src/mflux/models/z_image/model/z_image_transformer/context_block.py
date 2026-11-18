import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward


class ZImageContextBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, norm_eps: float = 1e-5, qk_norm: bool = True):
        super().__init__()
        self.attention = ZImageAttention(dim=dim, n_heads=n_heads, qk_norm=qk_norm, eps=1e-5)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(self, x: mx.array, attn_mask: mx.array | None, freqs_cis: mx.array) -> mx.array:
        attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + self.attention_norm2(attn_out)
        ffn_out = self.feed_forward(self.ffn_norm1(x))
        x = x + self.ffn_norm2(ffn_out)
        return x
