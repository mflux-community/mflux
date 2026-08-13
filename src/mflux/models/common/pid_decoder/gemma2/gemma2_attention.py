import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.gemma2.gemma2_config import Gemma2Config


class Gemma2Attention(nn.Module):
    def __init__(self, config: Gemma2Config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / (config.query_pre_attn_scalar**0.5)
        self.attn_logit_softcapping = config.attn_logit_softcapping

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)
        q = q * self.scale

        if self.repeats > 1:
            q = q.reshape(B, self.n_kv_heads, self.repeats, L, self.head_dim)
            k = mx.expand_dims(k, 2)
            v = mx.expand_dims(v, 2)

        scores = q @ k.swapaxes(-1, -2)
        scores = mx.tanh(scores / self.attn_logit_softcapping) * self.attn_logit_softcapping
        if mask is not None:
            # GQA reshapes scores to 5D (B, n_kv_heads, repeats, L, L). A 4D mask
            # (B, 1, L, L) right-aligns and aliases its batch axis with n_kv_heads
            # instead of B. Insert a singleton axis before the trailing (L, L)
            # dims to keep batch aligned with batch (mirrors mlx-vlm's
            # `align_attention_mask_to_scores`). No-op when repeats == 1.
            while mask.ndim < scores.ndim:
                mask = mx.expand_dims(mask, axis=max(mask.ndim - 2, 0))
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1, precise=True)
        out = scores @ v
        if self.repeats > 1:
            out = out.reshape(B, self.n_heads, L, self.head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)
