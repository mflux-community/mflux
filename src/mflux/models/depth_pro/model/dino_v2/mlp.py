import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int = 1024, hidden_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x
