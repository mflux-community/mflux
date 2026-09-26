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
        kv_pair: tuple[mx.array, mx.array] | None = None,
        kv_mode: str | None = None,
        prefix_len: int = 0,
        segments: list[tuple[int, int, bool, mx.array | None]] | None = None,
    ) -> mx.array | tuple[mx.array, tuple[mx.array, mx.array]]:
        scale1, gate1 = mx.split(mod1, 2, axis=-1)
        scale2, gate2 = mx.split(mod2, 2, axis=-1)

        attn_input = self.img_norm1(hidden_states) * (1 + scale1)
        attn_out = self.attn(
            attn_input, rope_cos, rope_sin, attn_mask, text_len, kv_pair, kv_mode, prefix_len, segments
        )
        if kv_mode is not None:
            attn_out, kv_pair = attn_out
        hidden_states = hidden_states + nn.tanh(gate1) * attn_out

        mlp_input = self.img_norm2(hidden_states) * (1 + scale2)
        hidden_states = hidden_states + nn.tanh(gate2) * self.img_mlp(mlp_input)
        if kv_mode is not None:
            return hidden_states, kv_pair
        return hidden_states
