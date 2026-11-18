import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward


class ZImageTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, norm_eps: float = 1e-5, qk_norm: bool = True):
        super().__init__()
        self.attention = ZImageAttention(dim=dim, n_heads=n_heads, qk_norm=qk_norm, eps=1e-5)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]

    def __call__(self, x: mx.array, attn_mask: mx.array | None, freqs_cis: mx.array, t_emb: mx.array) -> mx.array:
        # Compute modulation parameters
        modulation = mx.expand_dims(self.adaLN_modulation[0](t_emb), axis=1)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation, 4, axis=2)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        gate_msa = mx.tanh(gate_msa)
        gate_mlp = mx.tanh(gate_mlp)

        # Attention with modulation
        attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attention_norm2(attn_out)

        # FFN with modulation
        ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        x = x + gate_mlp * self.ffn_norm2(ffn_out)
        return x
