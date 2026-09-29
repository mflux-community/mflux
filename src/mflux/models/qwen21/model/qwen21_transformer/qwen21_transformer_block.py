import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_transformer.qwen21_attention import Qwen21Attention
from mflux.models.qwen21.model.qwen21_transformer.qwen21_feed_forward import Qwen21SwiGLUFeedForward


class Qwen21TransformerBlock(nn.Module):
    # Modulation is shared across blocks: the parent passes per-token modulation
    # chunks already selected for the causal_condition t=0 split.

    def __init__(
        self,
        dim: int = 4096,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn = Qwen21Attention(dim=dim, num_heads=num_attention_heads, head_dim=attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.img_mlp = Qwen21SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def __call__(
        self,
        hidden_states: mx.array,
        mod1: mx.array,
        mod2: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        attn_mask: mx.array | None,
        text_len: int | None = None,
    ) -> mx.array:
        scale1, gate1 = mx.split(mod1, 2, axis=-1)
        scale2, gate2 = mx.split(mod2, 2, axis=-1)

        attn_input = self.img_norm1(hidden_states) * (1 + scale1)
        hidden_states = hidden_states + nn.tanh(gate1) * self.attn(attn_input, rope_cos, rope_sin, attn_mask, text_len)

        mlp_input = self.img_norm2(hidden_states) * (1 + scale2)
        hidden_states = hidden_states + nn.tanh(gate2) * self.img_mlp(mlp_input)
        return hidden_states

    def text_forward(
        self,
        hidden_states: mx.array,
        mod1: mx.array,
        mod2: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Run this block over the text prefix only (causal, t=0 modulation).

        Returns the updated text hidden states and this block's post-rope K/V
        over the text positions, for reuse by the image-only denoise steps.
        """
        scale1, gate1 = mx.split(mod1, 2, axis=-1)
        scale2, gate2 = mx.split(mod2, 2, axis=-1)

        attn_input = self.img_norm1(hidden_states) * (1 + scale1)
        attn_out, key_text, value_text = self.attn.text_attention(attn_input, rope_cos, rope_sin)
        hidden_states = hidden_states + nn.tanh(gate1) * attn_out

        mlp_input = self.img_norm2(hidden_states) * (1 + scale2)
        hidden_states = hidden_states + nn.tanh(gate2) * self.img_mlp(mlp_input)
        return hidden_states, key_text, value_text

    def image_forward(
        self,
        hidden_states: mx.array,
        key_text: mx.array,
        value_text: mx.array,
        mod1: mx.array,
        mod2: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
    ) -> mx.array:
        """Run this block over the image tokens only, against the cached text K/V."""
        scale1, gate1 = mx.split(mod1, 2, axis=-1)
        scale2, gate2 = mx.split(mod2, 2, axis=-1)

        attn_input = self.img_norm1(hidden_states) * (1 + scale1)
        hidden_states = hidden_states + nn.tanh(gate1) * self.attn.image_attention(
            attn_input, key_text, value_text, rope_cos, rope_sin
        )

        mlp_input = self.img_norm2(hidden_states) * (1 + scale2)
        hidden_states = hidden_states + nn.tanh(gate2) * self.img_mlp(mlp_input)
        return hidden_states
