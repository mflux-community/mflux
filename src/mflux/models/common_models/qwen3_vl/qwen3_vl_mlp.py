from typing import Callable

import mlx.core as mx
from mlx import nn


class Qwen3VLMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, linear_class: Callable[..., nn.Module] = nn.Linear):
        super().__init__()
        self.gate_proj = linear_class(hidden_size, intermediate_size, bias=False)
        self.up_proj = linear_class(hidden_size, intermediate_size, bias=False)
        self.down_proj = linear_class(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate_output = nn.silu(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        intermediate_output = gate_output * up_output
        output = self.down_proj(intermediate_output)
        return output
