import mlx.core as mx
from mlx import nn

from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear


class Qwen3VLMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Fp8Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Fp8Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Fp8Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate_output = nn.silu(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        return self.down_proj(gate_output * up_output)
