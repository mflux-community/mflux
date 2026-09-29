from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_available,
)


class Qwen21Attention(nn.Module):
    def __init__(self, dim: int = 4096, num_heads: int = 32, head_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_out = [nn.Linear(num_heads * head_dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        # the fused kernel hard-codes eps = 1e-6. Other eps values use the composed path.
        self.use_fused_prologue = fused_qk_norm_rope_available(head_dim) and eps == 1e-6

    def __call__(
        self,
        hidden_states: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        attn_mask: mx.array | None,
        text_len: int | None = None,
    ) -> mx.array:
        query, key, value = self._project(hidden_states, rope_cos, rope_sin)

        if text_len is not None and attn_mask is None:
            # block-causal, segmented like the reference processor: causal text attention
            # plus a maskless full-attention call for the target block (fast paths both)
            text_out = scaled_dot_product_attention(
                query[:, :, :text_len],
                key[:, :, :text_len],
                value[:, :, :text_len],
                scale=self.head_dim**-0.5,
                mask="causal",
            )
            target_out = scaled_dot_product_attention(
                query[:, :, text_len:],
                key,
                value,
                scale=self.head_dim**-0.5,
            )
            return self._unproject(mx.concatenate([text_out, target_out], axis=2))
        hidden_states = scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.head_dim**-0.5,
            mask=attn_mask,
        )
        return self._unproject(hidden_states)

    def _project(
        self,
        hidden_states: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Shared Q/K/V projection: returns [batch, heads, seq, head_dim]."""
        if self.use_fused_prologue:
            fused = fused_qk_norm_rope(
                self.to_q(hidden_states),
                self.to_k(hidden_states),
                self.norm_q.weight,
                self.norm_k.weight,
                rope_cos,
                rope_sin,
                self.num_heads,
                self.head_dim,
            )
            if fused is not None:
                out_q, out_k = fused
                batch, length = out_q.shape[:2]
                query = mx.transpose(out_q.reshape(batch, length, self.num_heads, self.head_dim), (0, 2, 1, 3))
                key = mx.transpose(out_k.reshape(batch, length, self.num_heads, self.head_dim), (0, 2, 1, 3))
                value = mx.transpose(
                    mx.reshape(self.to_v(hidden_states), (*hidden_states.shape[:-1], self.num_heads, self.head_dim)),
                    (0, 2, 1, 3),
                )
                return query, key, value

        query = mx.reshape(self.to_q(hidden_states), (*hidden_states.shape[:-1], self.num_heads, self.head_dim))
        key = mx.reshape(self.to_k(hidden_states), (*hidden_states.shape[:-1], self.num_heads, self.head_dim))
        value = mx.reshape(self.to_v(hidden_states), (*hidden_states.shape[:-1], self.num_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = Qwen21Attention._apply_rope(query, rope_cos, rope_sin)
        key = Qwen21Attention._apply_rope(key, rope_cos, rope_sin)

        return mx.transpose(query, (0, 2, 1, 3)), mx.transpose(key, (0, 2, 1, 3)), mx.transpose(value, (0, 2, 1, 3))

    def _unproject(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(hidden_states, (*hidden_states.shape[:-2], self.num_heads * self.head_dim))
        return self.to_out[0](hidden_states)

    def text_attention(
        self,
        hidden_states: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Causal text-prefix attention; also returns the prefix K/V to cache.

        Text tokens read the timestep-independent t=0 modulation row and only
        attend (causally) to each other, so their post-rope K/V are identical on
        every denoise step (and independent of the image stream) — compute them
        once per prompt and reuse them for the image-only steps.
        """
        query, key, value = self._project(hidden_states, rope_cos, rope_sin)
        hidden_states = scaled_dot_product_attention(query, key, value, scale=self.head_dim**-0.5, mask="causal")
        return self._unproject(hidden_states), mx.contiguous(key), mx.contiguous(value)

    def image_attention(
        self,
        hidden_states: mx.array,
        key_text: mx.array,
        value_text: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
    ) -> mx.array:
        """Image-block attention over the full [cached text | image] sequence."""
        query, key, value = self._project(hidden_states, rope_cos, rope_sin)
        key = mx.concatenate([key_text, key], axis=2)
        value = mx.concatenate([value_text, value], axis=2)
        hidden_states = scaled_dot_product_attention(query, key, value, scale=self.head_dim**-0.5)
        return self._unproject(hidden_states)

    @staticmethod
    def _apply_rope(x: mx.array, cos_vals: mx.array, sin_vals: mx.array) -> mx.array:
        # Real form of the reference's complex-multiplication rope: pairs (2k, 2k+1) share angle k.
        x_float = x.astype(mx.float32)
        x_reshaped = mx.reshape(x_float, (*x.shape[:-1], -1, 2))
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]
        freqs_cos = cos_vals[None, :, None, :]
        freqs_sin = sin_vals[None, :, None, :]
        out_real = x_real * freqs_cos - x_imag * freqs_sin
        out_imag = x_real * freqs_sin + x_imag * freqs_cos
        out_pairs = mx.stack([out_real, out_imag], axis=-1)
        x_out = mx.reshape(out_pairs, (*x.shape[:-1], -1))
        return x_out.astype(x.dtype)
