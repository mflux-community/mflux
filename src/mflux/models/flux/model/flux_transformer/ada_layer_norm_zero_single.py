import mlx.core as mx
from mlx import nn


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(self, dim: int = 3072):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 3 * dim)
        self.norm = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)

    def __call__(self, hidden_states: mx.array, text_embeddings: mx.array) -> mx.array:
        text_embeddings = self.linear(nn.silu(text_embeddings))
        chunk_size = self.dim
        shift_msa = text_embeddings[:, 0 * chunk_size : 1 * chunk_size]
        scale_msa = text_embeddings[:, 1 * chunk_size : 2 * chunk_size]
        gate_msa = text_embeddings[:, 2 * chunk_size : 3 * chunk_size]
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa
