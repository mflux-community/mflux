import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm
from mflux.models.krea2.model.krea2_transformer.modulation import SimpleModulation


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = Krea2RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def __call__(self, x: mx.array, tvec: mx.array) -> mx.array:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)
