from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


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

    def __call__(
        self,
        hidden_states: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        attn_mask: mx.array | None,
        text_len: int | None = None,
        kv_pair: tuple[mx.array, mx.array] | None = None,
        kv_mode: str | None = None,
        prefix_len: int = 0,
        segments: list[tuple[int, int, bool, mx.array | None]] | None = None,
    ) -> mx.array | tuple[mx.array, tuple[mx.array, mx.array]]:
        query = mx.reshape(self.to_q(hidden_states), (*hidden_states.shape[:-1], self.num_heads, self.head_dim))
        key = mx.reshape(self.to_k(hidden_states), (*hidden_states.shape[:-1], self.num_heads, self.head_dim))
        value = mx.reshape(self.to_v(hidden_states), (*hidden_states.shape[:-1], self.num_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = Qwen21Attention._apply_rope(query, rope_cos, rope_sin)
        key = Qwen21Attention._apply_rope(key, rope_cos, rope_sin)

        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))

        if segments is not None:
            # exact multi-pass prefill, one call per run: a text run attends its whole
            # prefix causally (small additive mask), an image run attends its whole
            # prefix unmasked. Avoids any quadratic dense mask tensor.
            out_parts = []
            for start, end, is_text, seg_mask in segments:
                out_parts.append(
                    scaled_dot_product_attention(
                        query[:, :, start:end],
                        key[:, :, :end],
                        value[:, :, :end],
                        scale=self.head_dim**-0.5,
                        mask=seg_mask,
                    )
                )
            attn_out = self._merge(mx.concatenate(out_parts, axis=2))
            if kv_mode == "extract":
                prefix_kv = (key[:, :, :prefix_len], value[:, :, :prefix_len])
                return attn_out, prefix_kv
            return attn_out

        if kv_mode == "extract":
            # dense-mask fallback; store the prefix (text + reference) K/V --
            # causal_condition keeps them step-independent
            attn_out = scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.head_dim**-0.5,
                mask=attn_mask,
            )
            prefix_kv = (key[:, :, :prefix_len], value[:, :, :prefix_len])
            return self._merge(attn_out), prefix_kv

        if kv_mode == "cached":
            # later steps: only target queries are recomputed; they attend the frozen
            # prefix K/V from the cache plus their own tokens -- full attention, no mask
            key = mx.concatenate([kv_pair[0], key], axis=2)
            value = mx.concatenate([kv_pair[1], value], axis=2)
            attn_out = scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.head_dim**-0.5,
            )
            return self._merge(attn_out), kv_pair

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
            hidden_states = mx.concatenate([text_out, target_out], axis=2)
        else:
            hidden_states = scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.head_dim**-0.5,
                mask=attn_mask,
            )
        return self._merge(hidden_states)

    def _merge(self, attn_out: mx.array) -> mx.array:
        hidden_states = mx.transpose(attn_out, (0, 2, 1, 3))
        hidden_states = mx.reshape(hidden_states, (*hidden_states.shape[:-2], self.num_heads * self.head_dim))
        return self.to_out[0](hidden_states)

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
