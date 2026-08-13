import mlx.core as mx
from mlx import nn


class Gemma2RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Gemma weights are stored as (weight - 1), matching mlx-vlm's convention
        # (`mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)`) and HF's
        # `Gemma2RMSNorm.forward` (`output * (1.0 + self.weight.float())`).
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
