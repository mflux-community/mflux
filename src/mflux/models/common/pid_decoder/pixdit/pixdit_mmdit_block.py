import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.pixdit.pixdit_attention import flash_sdpa
from mflux.models.common.pid_decoder.pixdit.pixdit_feed_forward import PixDiTFeedForward
from mflux.models.common.pid_decoder.pixdit.pixdit_rms_norm import PixDiTRMSNorm
from mflux.models.common.pid_decoder.pixdit.pixdit_rope import apply_rotary_emb


def _apply_adaln(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1.0 + scale) + shift


class MMDiTJointAttention(nn.Module):
    """Source: pixeldit_official.py MMDiTJointAttention (CP branches dropped)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_x = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm_x = PixDiTRMSNorm(self.head_dim)
        self.k_norm_x = PixDiTRMSNorm(self.head_dim)
        self.q_norm_y = PixDiTRMSNorm(self.head_dim)
        self.k_norm_y = PixDiTRMSNorm(self.head_dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

    def __call__(self, x, y, pos_img, pos_txt, attn_mask):
        B, Nx, C = x.shape
        _, Ny, _ = y.shape
        H, Hc = self.num_heads, self.head_dim

        # reshape(B, N, 3, H, Hc) then indexing axis 2 directly is equivalent to
        # PyTorch's reshape(...).permute(2, 0, 1, 3, 4)[i] -- both isolate the
        # "3" axis right after batch/seq, before heads.
        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, H, Hc)
        qx, kx, vx = qkv_x[:, :, 0], qkv_x[:, :, 1], qkv_x[:, :, 2]
        qx, kx = self.q_norm_x(qx), self.k_norm_x(kx)

        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, H, Hc)
        qy, ky, vy = qkv_y[:, :, 0], qkv_y[:, :, 1], qkv_y[:, :, 2]
        qy, ky = self.q_norm_y(qy), self.k_norm_y(ky)

        # RoPE applied while still [B, N, H, Hc] -- apply_rotary_emb broadcasts
        # freqs over axis 1 (seq), which is only correct before the transpose below.
        qx, kx = apply_rotary_emb(qx, kx, pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, pos_txt)

        qx, kx, vx = (t.transpose(0, 2, 1, 3) for t in (qx, kx, vx))
        qy, ky, vy = (t.transpose(0, 2, 1, 3) for t in (qy, ky, vy))

        # Joint sequence order is [text, image] (matches source's cat([qy, qx])).
        q = mx.concatenate([qy, qx], axis=2)
        k = mx.concatenate([ky, kx], axis=2)
        v = mx.concatenate([vy, vx], axis=2)

        scale = Hc**-0.5
        out = flash_sdpa(q, k, v, scale=scale, mask=attn_mask)

        out_y = out[:, :, :Ny, :].transpose(0, 2, 1, 3).reshape(B, Ny, C)
        out_x = out[:, :, Ny:, :].transpose(0, 2, 1, 3).reshape(B, Nx, C)
        return self.proj_x(out_x), self.proj_y(out_y)


class MMDiTBlockT2I(nn.Module):
    """Source: pixeldit_official.py MMDiTBlockT2I (CP branches, dropout dropped)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_x1 = PixDiTRMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = PixDiTRMSNorm(hidden_size, eps=1e-6)
        self.attn = MMDiTJointAttention(hidden_size, num_heads)
        self.norm_x2 = PixDiTRMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = PixDiTRMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = PixDiTFeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = PixDiTFeedForward(hidden_size, mlp_hidden_dim)
        # Single-element list mirrors the source's nn.Sequential(nn.Linear(...))
        # wrapper -- MLX has no nn.Sequential -- so Task 9's weight-mapping key
        # "adaLN_modulation_img.0.weight" maps to adaLN_modulation_img[0].weight.
        self.adaLN_modulation_img = [nn.Linear(hidden_size, 6 * hidden_size, bias=True)]
        self.adaLN_modulation_txt = [nn.Linear(hidden_size, 6 * hidden_size, bias=True)]

    def __call__(self, x, y, c, pos_img, pos_txt=None, attn_mask=None):
        img_params = self.adaLN_modulation_img[0](c)
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = mx.split(img_params, 6, axis=-1)
        txt_params = self.adaLN_modulation_txt[0](c)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = mx.split(txt_params, 6, axis=-1)

        x_norm = _apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = _apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        x = x + gate_mlp_x * self.mlp_x(_apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(_apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y
