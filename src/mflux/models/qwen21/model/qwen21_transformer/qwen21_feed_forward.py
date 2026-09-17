import mlx.core as mx
from mlx import nn


class Qwen21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.out(nn.silu(self.gate_layer(hidden_states)) * self.proj(hidden_states))
