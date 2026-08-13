import mlx.core as mx
from mlx import nn

from mflux.models.ernie_image.model.ernie_transformer.attention import ErnieAttention
from mflux.models.ernie_image.model.ernie_transformer.feed_forward import ErnieFeedForward


class ErnieTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, eps: float, qk_layernorm: bool):
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieAttention(hidden_size, num_heads, eps, qk_layernorm)
        self.adaLN_mlp_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieFeedForward(hidden_size, ffn_hidden_size)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        temb: tuple,
        mask: mx.array | None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        residual = x
        x = self.adaLN_sa_ln(x)
        x = x * (1 + scale_msa) + shift_msa
        x = residual + gate_msa * self.self_attention(x, cos, sin, mask)

        residual = x
        x = self.adaLN_mlp_ln(x)
        x = x * (1 + scale_mlp) + shift_mlp
        x = residual + gate_mlp * self.mlp(x)

        return x
