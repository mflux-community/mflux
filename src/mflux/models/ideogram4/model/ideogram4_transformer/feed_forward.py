import mlx.core as mx
from mlx import nn

from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = Fp8Linear(dim, hidden_dim, bias=False)
        self.w2 = Fp8Linear(hidden_dim, dim, bias=False)
        self.w3 = Fp8Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))
