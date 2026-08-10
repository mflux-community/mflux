import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.attention import Krea2Attention
from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm
from mflux.models.krea2.model.krea2_transformer.feed_forward import Krea2SwiGLU


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int | None = None):
        super().__init__()
        self.prenorm = Krea2RMSNorm(features)
        self.postnorm = Krea2RMSNorm(features)
        self.attn = Krea2Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = Krea2SwiGLU(features, multiplier, bias)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.prenorm(x), freqs=None, mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int | None = None,
    ):
        super().__init__()
        self.layerwise_blocks = [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        b, s, n, d = x.shape  # (B, seq, txtlayers, txtdim)
        x = x.reshape(b * s, n, d)
        for block in self.layerwise_blocks:
            x = block(x, mask=None)
        # (b*s, n, d) -> (b, s, d, n), project layers n -> 1, squeeze
        x = x.reshape(b, s, n, d).transpose(0, 1, 3, 2)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x
