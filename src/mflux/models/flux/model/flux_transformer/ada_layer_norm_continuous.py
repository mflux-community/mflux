import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, bias: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        # The FLUX.1, FIBO and Qwen-Image checkpoints carry a bias for this layer; FLUX.2 has none.
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if bias:
            # Zero until the checkpoint sets it: mflux 0.16.0 to 0.21.0 dropped this bias, so a model
            # those versions saved has none, and it then loads the way it was saved. In the model's
            # precision, or adding it would promote the stream to float32.
            self.linear.bias = mx.zeros((embedding_dim * 2,), dtype=ModelConfig.precision)
        self.norm = nn.LayerNorm(dims=embedding_dim, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        text_embeddings = self.linear(nn.silu(text_embeddings).astype(ModelConfig.precision))
        chunk_size = self.embedding_dim
        scale = text_embeddings[:, 0 * chunk_size : 1 * chunk_size]
        shift = text_embeddings[:, 1 * chunk_size : 2 * chunk_size]
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
