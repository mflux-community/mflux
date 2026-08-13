import mlx.core as mx
from mlx import nn

from mflux.models.ideogram4.model.ideogram4_transformer.attention import Ideogram4Attention
from mflux.models.ideogram4.model.ideogram4_transformer.feed_forward import Ideogram4MLP
from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear
from mflux.models.ideogram4.model.ideogram4_transformer.modulation import Ideogram4RMSNorm


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)
        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.adaln_modulation = Fp8Linear(adanln_dim, 4 * hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array,
        cos: mx.array,
        sin: mx.array,
        adaln_input: mx.array,
    ) -> mx.array:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(mod, 4, axis=-1)
        gate_msa = mx.tanh(gate_msa)
        gate_mlp = mx.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        return x + gate_mlp * self.ffn_norm2(ffn_out)
