import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.pixdit.pixdit_sincos_pos_embed import get_2d_sincos_pos_embed

# Source: pixeldit_official.py PatchTokenEmbedder / PixelTokenEmbedder (lines 357-442).
# PixelTokenEmbedder's "legacy patch mode" (3D [B*L, P2, C] input) is dropped -- PidNet
# only ever calls it in "image mode" ([B, C, H, W] input), same pattern as the CP-branch
# drops elsewhere in this port.


class PatchTokenEmbedder(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, norm=None):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=True)
        self.norm = norm if norm is not None else (lambda x: x)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.proj(x))


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_output: int):
        super().__init__()
        self.hidden_size_output = hidden_size_output
        self.proj = nn.Linear(in_channels, hidden_size_output, bias=True)

    def __call__(self, inputs: mx.array, img_height: int, img_width: int, patch_size: int) -> mx.array:
        # inputs: [B, C, H, W] (mflux channel-first) -> NHWC for the per-pixel linear proj.
        B, C, H, W = inputs.shape
        Hs, Ws = H // patch_size, W // patch_size
        P2 = patch_size * patch_size

        x = inputs.transpose(0, 2, 3, 1)  # [B, H, W, C]
        x = self.proj(x)  # [B, H, W, D]

        pos_full = get_2d_sincos_pos_embed(self.hidden_size_output, H, W)  # [H*W, D]
        pos_full = pos_full.reshape(H, W, self.hidden_size_output)
        x = x + pos_full[None]

        x = x.reshape(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # [B, Hs, Ws, ps, ps, D]
        return x.reshape(B * Hs * Ws, P2, self.hidden_size_output)
