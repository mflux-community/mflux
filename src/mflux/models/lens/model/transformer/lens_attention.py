import mlx.core as mx
from mlx import nn


def _apply_rope(x: mx.array, cos_vals: mx.array, sin_vals: mx.array) -> mx.array:
    # Interleaved-pair rotation on [B, S, H, D], freqs indexed by sequence.
    x_float = x.astype(mx.float32)
    x_reshaped = mx.reshape(x_float, (*x.shape[:-1], -1, 2))
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    freqs_cos = cos_vals[None, :, None, :]
    freqs_sin = sin_vals[None, :, None, :]
    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos
    out_pairs = mx.stack([out_real, out_imag], axis=-1)
    return mx.reshape(out_pairs, (*x.shape[:-1], -1)).astype(x.dtype)


class LensJointAttention(nn.Module):
    """Joint image+text attention, fused QKV per stream, image first in the joint
    sequence (the reference order; Flux and Qwen put text first)."""

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm_q = nn.RMSNorm(head_dim, eps=1e-5)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-5)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-5)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-5)

        self.img_qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.txt_qkv = nn.Linear(dim, 3 * dim, bias=True)
        # List so the weight path is to_out.0, matching the checkpoint.
        self.to_out = [nn.Linear(dim, dim, bias=True)]
        self.to_add_out = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        text_rotary_emb: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        batch, seq_img, _ = hidden_states.shape
        seq_txt = encoder_hidden_states.shape[1]

        img_qkv = self.img_qkv(hidden_states).reshape(batch, seq_img, 3, self.num_heads, self.head_dim)
        img_q = self.norm_q(img_qkv[:, :, 0])
        img_k = self.norm_k(img_qkv[:, :, 1])
        img_v = img_qkv[:, :, 2]

        txt_qkv = self.txt_qkv(encoder_hidden_states).reshape(batch, seq_txt, 3, self.num_heads, self.head_dim)
        txt_q = self.norm_added_q(txt_qkv[:, :, 0])
        txt_k = self.norm_added_k(txt_qkv[:, :, 1])
        txt_v = txt_qkv[:, :, 2]

        img_cos, img_sin = image_rotary_emb
        txt_cos, txt_sin = text_rotary_emb
        img_q = _apply_rope(img_q, img_cos, img_sin)
        img_k = _apply_rope(img_k, img_cos, img_sin)
        txt_q = _apply_rope(txt_q, txt_cos, txt_sin)
        txt_k = _apply_rope(txt_k, txt_cos, txt_sin)

        query = mx.concatenate([img_q, txt_q], axis=1)
        key = mx.concatenate([img_k, txt_k], axis=1)
        value = mx.concatenate([img_v, txt_v], axis=1)

        q_bhsd = mx.transpose(query, (0, 2, 1, 3))
        k_bhsd = mx.transpose(key, (0, 2, 1, 3))
        v_bhsd = mx.transpose(value, (0, 2, 1, 3))
        out = mx.fast.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, scale=1.0 / (self.head_dim**0.5), mask=None)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, seq_img + seq_txt, -1)
        out = out.astype(hidden_states.dtype)

        img_out = self.to_out[0](out[:, :seq_img])
        txt_out = self.to_add_out(out[:, seq_img:])
        return img_out, txt_out
