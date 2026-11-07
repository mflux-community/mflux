import mlx.core as mx
from mlx import nn

from mflux.models.common_models.aiartlab_adapter.adapter_attention_block import AdapterAttentionBlock


class AdapterAttentionMixer(nn.Module):
    # Token attention branch of the AiArtLab text adapters: hidden-state slices (B, L, in_dim) -> (B, L, out_dim).
    def __init__(self, in_dim: int, out_dim: int, dim: int, heads: int, blocks: int, max_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-5)
        self.inp = nn.Linear(in_dim, dim)
        self.pos = mx.zeros((1, max_len, dim))
        self.blocks = [AdapterAttentionBlock(dim, heads, 4) for _ in range(blocks)]
        self.out = nn.Linear(dim, out_dim)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # mask: (B, L), true for real tokens. Without a mask every token attends to every token.
        h = self.inp(self.norm(x))
        length = h.shape[1]
        pos = self.pos[:, :length]
        if length > pos.shape[1]:
            # Positions past the trained table get no positional signal, as in the reference.
            pos = mx.pad(pos, [(0, 0), (0, length - pos.shape[1]), (0, 0)])
        h = h + pos
        attention_mask = None
        if mask is not None and not bool(mx.all(mask)):
            attention_mask = mx.where(mask.astype(mx.bool_)[:, None, None, :], 0.0, -mx.inf).astype(h.dtype)
        for block in self.blocks:
            h = block(h, attention_mask)
        return self.out(h)
