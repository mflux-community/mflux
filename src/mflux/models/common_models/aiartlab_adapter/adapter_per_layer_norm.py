import mlx.core as mx
from mlx import nn


class AdapterPerLayerNorm(nn.Module):
    # AiArtLab "ln-per-layer": a plain layer norm over each hidden-state slice, then an affine one over all.
    def __init__(self, in_dim: int, n_slices: int):
        super().__init__()
        self.n_slices = n_slices
        self.slice_dim = in_dim // n_slices
        self.all = nn.LayerNorm(in_dim, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        x = mx.fast.layer_norm(x.reshape(batch, length, self.n_slices, self.slice_dim), None, None, 1e-5)
        return self.all(x.reshape(batch, length, -1))
