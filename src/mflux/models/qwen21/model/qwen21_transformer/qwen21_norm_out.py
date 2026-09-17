import mlx.core as mx
from mlx import nn


class Qwen21AdaLayerNormContinuous(nn.Module):
    # Final adaptive norm, scale-only: no shift, so the linear maps to embedding_dim.

    def __init__(self, embedding_dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=False)

    def __call__(self, hidden_states: mx.array, scale: mx.array) -> mx.array:
        return self.norm(hidden_states) * (1 + scale)
