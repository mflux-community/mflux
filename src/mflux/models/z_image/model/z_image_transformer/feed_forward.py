import mlx.core as mx
from mlx import nn

from mflux.models.common.compute_precision import ComputePrecision


class FeedForward(nn.Module):
    # In float16 the SwiGLU product (~3e5) and the output (~1e6) overflow, so the product is computed
    # this many times smaller and the output scaled back in the stream dtype. No biases: exact up to rounding.
    FLOAT16_HEADROOM = 32.0

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.compute_precision = ComputePrecision()

    def __call__(self, x: mx.array) -> mx.array:
        precision = self.compute_precision
        dtype = x.dtype
        x = precision.to_compute(x)
        hidden = nn.silu(self.w1(x)) * precision.shrink(self.w3(x), FeedForward.FLOAT16_HEADROOM)
        return precision.to_stream(self.w2(hidden), dtype, FeedForward.FLOAT16_HEADROOM)
