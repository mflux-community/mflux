import math

import mlx.core as mx
from mlx import nn


class SigmaAwarePerTokenGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.content_proj = nn.Linear(dim * 2, 1)
        self.log_alpha = mx.array(math.log(5.0))

    def __call__(self, x: mx.array, lq: mx.array, sigma: mx.array) -> mx.array:
        content_logit = self.content_proj(mx.concatenate([x, lq], axis=-1))  # [B, N, 1]
        sigma_offset = -mx.exp(self.log_alpha) * sigma.astype(mx.float32).reshape(-1, 1, 1)
        gate = mx.sigmoid(content_logit + sigma_offset)
        return x + gate * lq


class SigmaAwarePerTokenAndDimGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.content_proj = nn.Linear(dim * 2, dim)
        self.log_alpha = mx.array(math.log(5.0))

    def __call__(self, x: mx.array, lq: mx.array, sigma: mx.array) -> mx.array:
        content_logit = self.content_proj(mx.concatenate([x, lq], axis=-1))  # [B, N, D]
        sigma_offset = -mx.exp(self.log_alpha) * sigma.astype(mx.float32).reshape(-1, 1, 1)
        gate = mx.sigmoid(content_logit + sigma_offset)
        return x + gate * lq


def build_gate(gate_type: str, dim: int) -> nn.Module:
    if gate_type == "sigma_aware_per_token":
        return SigmaAwarePerTokenGate(dim)
    if gate_type == "sigma_aware_per_token_per_dim":
        return SigmaAwarePerTokenAndDimGate(dim)
    raise ValueError(f"Unknown gate_type: {gate_type!r}")
