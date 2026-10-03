import mlx.core as mx
from mlx import nn

from mflux.models.common.compute_precision import ComputePrecision


class Qwen21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.compute_precision = ComputePrecision()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        dtype = hidden_states.dtype
        hidden_states = self.compute_precision.to_compute(hidden_states)
        hidden_states = self.out(nn.silu(self.gate_layer(hidden_states)) * self.proj(hidden_states))
        return self.compute_precision.to_stream(hidden_states, dtype)
