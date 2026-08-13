import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm


class Krea2TextMLP(nn.Module):
    def __init__(self, txtdim: int, features: int):
        super().__init__()
        self.norm = Krea2RMSNorm(txtdim)
        self.linear_in = nn.Linear(txtdim, features)
        self.linear_out = nn.Linear(features, features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_out(nn.gelu_approx(self.linear_in(self.norm(x))))
