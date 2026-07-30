import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.pixdit.pixdit_attention import flash_sdpa
from mflux.models.common.pid_decoder.pixdit.pixdit_mmdit_block import _apply_adaln
from mflux.models.common.pid_decoder.pixdit.pixdit_rms_norm import PixDiTRMSNorm
from mflux.models.common.pid_decoder.pixdit.pixdit_rope import apply_rotary_emb, precompute_freqs_cis_2d_ntk

# Source: pixeldit_official.py RotaryAttention / MLP / PiTBlock (lines 248-542).
# CP branches, dropout, and the non-NTK "original" rope_mode (never selected for the
# ported checkpoint -- Task 4 only ported precompute_freqs_cis_2d_ntk) are dropped.


class PiTMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x):
        # Source's MLP.act is nn.GELU() (exact, erf-based) -- use mx's exact
        # nn.gelu, not the tanh approximation nn.gelu_approx.
        return self.fc2(nn.gelu(self.fc1(x)))


class PiTRotaryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = PixDiTRMSNorm(self.head_dim)
        self.k_norm = PixDiTRMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, pos, mask=None) -> mx.array:
        B, N, C = x.shape
        H, Hc = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, Hc)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb(q, k, pos)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        out = flash_sdpa(q, k, v, scale=Hc**-0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        attn_hidden_size: int,
        attn_num_heads: int,
        rope_ref_grid_h: int,
        rope_ref_grid_w: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.pixel_dim = pixel_hidden_size
        self.patch_size = patch_size
        # attn_dim is the PiT block's *internal* attention width -- distinct from
        # pixel_dim (the per-pixel stream's own hidden size). For the ported
        # checkpoint pixel_dim=16, attn_dim=1152: compress_to_attn/expand_from_attn
        # below are what bridge the two, never assume they're equal.
        self.attn_dim = attn_hidden_size
        self.num_heads = attn_num_heads
        self.rope_ref_grid_h = rope_ref_grid_h
        self.rope_ref_grid_w = rope_ref_grid_w
        p2 = patch_size * patch_size

        self.compress_to_attn = nn.Linear(p2 * pixel_hidden_size, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * pixel_hidden_size, bias=True)
        self.norm1 = PixDiTRMSNorm(pixel_hidden_size, eps=1e-6)
        self.attn = PiTRotaryAttention(self.attn_dim, self.num_heads)
        self.norm2 = PixDiTRMSNorm(pixel_hidden_size, eps=1e-6)
        self.mlp = PiTMLP(pixel_hidden_size, mlp_ratio)
        # Single-element list mirrors the source's nn.Sequential(nn.Linear(...))
        # wrapper -- MLX has no nn.Sequential -- so Task 9's weight-mapping key
        # "adaLN_modulation.0.weight" maps to adaLN_modulation[0].weight.
        self.adaLN_modulation = [nn.Linear(patch_hidden_size, 6 * pixel_hidden_size * p2, bias=True)]

    def __call__(
        self,
        x: mx.array,
        s_cond: mx.array,
        image_height: int,
        image_width: int,
        patch_size: int,
        mask=None,
    ) -> mx.array:
        BL, P2, C = x.shape
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        B = BL // L

        cond_params = self.adaLN_modulation[0](s_cond).reshape(BL, P2, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(cond_params, 6, axis=-1)

        x_norm = _apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.reshape(BL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).reshape(B, L, self.attn_dim)

        head_dim = self.attn_dim // self.num_heads
        pos = precompute_freqs_cis_2d_ntk(head_dim, Hs, Ws, self.rope_ref_grid_h, self.rope_ref_grid_w)
        attn_out = self.attn(x_comp, pos, mask)  # [B, L, attn_dim]
        attn_flat = self.expand_from_attn(attn_out.reshape(B * L, self.attn_dim))
        attn_exp = attn_flat.reshape(BL, P2, self.pixel_dim)

        x = x + gate_msa * attn_exp
        mlp_out = self.mlp(_apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_out
        return x
