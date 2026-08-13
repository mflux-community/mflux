import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.attention import Krea2Attention
from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm
from mflux.models.krea2.model.krea2_transformer.feed_forward import Krea2SwiGLU
from mflux.models.krea2.model.krea2_transformer.modulation import DoubleSharedModulation


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = Krea2RMSNorm(features)
        self.postnorm = Krea2RMSNorm(features)
        self.attn = Krea2Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = Krea2SwiGLU(features, multiplier, bias)

    def __call__(self, x: mx.array, vec: mx.array, freqs: mx.array, mask: mx.array | None = None) -> mx.array:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs=freqs, mask=mask)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x
