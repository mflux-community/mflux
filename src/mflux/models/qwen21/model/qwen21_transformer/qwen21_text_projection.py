import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_transformer.qwen21_zero_center_rms_norm import Qwen21ZeroCenterRMSNorm


class Qwen21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int = 4096, hidden_size: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.text_norm = Qwen21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)
        return self.out_layer(hidden_states)
