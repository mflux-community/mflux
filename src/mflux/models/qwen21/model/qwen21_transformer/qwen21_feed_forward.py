import mlx.core as mx
from mlx import nn

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.quantization.w8a8 import W8A8Linear


class Qwen21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.compute_precision = ComputePrecision()
        self._gate_proj_weight = None
        self._w8a8_gate_proj = None  # (weight, scale, sizes) when gate/proj are W8A8 (Qwen21W8A8.load)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        dtype = hidden_states.dtype
        hidden_states = self.compute_precision.to_compute(hidden_states)
        if self._w8a8_gate_proj is not None:
            gate, proj = W8A8Linear.fused(hidden_states, self._w8a8_gate_proj)
        elif self._gate_proj_weight is None:
            gate, proj = self.gate_layer(hidden_states), self.proj(hidden_states)
        else:
            gate, proj = mx.split(hidden_states @ self._gate_proj_weight.T, 2, axis=-1)
        hidden_states = self.out(nn.silu(gate) * proj)
        return self.compute_precision.to_stream(hidden_states, dtype)
