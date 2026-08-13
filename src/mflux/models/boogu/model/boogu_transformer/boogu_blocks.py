from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.boogu.model.boogu_transformer.boogu_attention import (
    BooguAttention,
    BooguDoubleStreamJointAttention,
)
from mflux.models.boogu.model.boogu_transformer.boogu_embeddings import (
    LuminaFeedForward,
    LuminaRMSNormZero,
)
from mflux.models.boogu.model.boogu_transformer.boogu_rope import RotaryEmb


class BooguImageTransformerBlock(nn.Module):
    """Base Boogu block: GQA self-attention + SwiGLU FFN with RMSNorm.

    With ``modulation=True`` (refiner / single-stream layers) it uses
    ``LuminaRMSNormZero`` and ``tanh``-gated residuals driven by ``temb``. With
    ``modulation=False`` (context refiner) it is a plain pre-norm block.

    Args:
        dim: Model dimension.
        num_attention_heads: Query head count.
        num_kv_heads: Key/value head count.
        multiple_of: FFN intermediate rounding.
        ffn_dim_multiplier: Optional FFN multiplier.
        norm_eps: RMSNorm epsilon.
        modulation: Whether to apply timestep modulation.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.modulation = modulation
        self.attn = BooguAttention(dim, num_attention_heads, num_kv_heads, eps=1e-5)
        self.feed_forward = LuminaFeedForward(dim, 4 * dim, multiple_of, ffn_dim_multiplier)

        if modulation:
            self.norm1 = LuminaRMSNormZero(dim, norm_eps)
        else:
            self.norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_emb: RotaryEmb,
        temb: mx.array | None = None,
    ) -> mx.array:
        if self.modulation:
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(norm_hidden_states, rotary_emb)
            hidden_states = hidden_states + mx.tanh(gate_msa)[:, None] * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp[:, None]))
            hidden_states = hidden_states + mx.tanh(gate_mlp)[:, None] * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(norm_hidden_states, rotary_emb)
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)
        return hidden_states


class BooguImageDoubleStreamTransformerBlock(nn.Module):
    """Double-stream (dual-stream) block: parallel image and instruction streams.

    The image stream runs a joint image<->instruction attention plus an image
    self-attention; the instruction stream runs the cross-attention contribution
    plus its own FFN. All residuals are ``tanh``-gated by ``temb`` modulation.

    Args:
        dim: Model dimension.
        num_attention_heads: Query head count.
        num_kv_heads: Key/value head count.
        multiple_of: FFN intermediate rounding.
        ffn_dim_multiplier: Optional FFN multiplier.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = dim

        self.img_instruct_attn = BooguDoubleStreamJointAttention(dim, num_attention_heads, num_kv_heads, eps=1e-5)
        self.img_self_attn = BooguAttention(dim, num_attention_heads, num_kv_heads, eps=1e-5)
        self.img_feed_forward = LuminaFeedForward(dim, 4 * dim, multiple_of, ffn_dim_multiplier)

        self.img_norm1 = LuminaRMSNormZero(dim, norm_eps)
        self.img_norm2 = LuminaRMSNormZero(dim, norm_eps)
        self.img_norm3 = LuminaRMSNormZero(dim, norm_eps)
        self.img_ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.img_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.img_self_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

        self.instruct_feed_forward = LuminaFeedForward(dim, 4 * dim, multiple_of, ffn_dim_multiplier)
        self.instruct_norm1 = LuminaRMSNormZero(dim, norm_eps)
        self.instruct_norm2 = LuminaRMSNormZero(dim, norm_eps)
        self.instruct_ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.instruct_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        img_hidden_states: mx.array,
        instruct_hidden_states: mx.array,
        image_rotary_emb: RotaryEmb,
        joint_rotary_emb: RotaryEmb,
        temb: mx.array,
    ) -> tuple[mx.array, mx.array]:
        l_instruct = instruct_hidden_states.shape[1]

        # Modulation for both streams.
        img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(img_hidden_states, temb)
        img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
        img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

        instruct_norm1_out, instruct_gate_msa, instruct_scale_mlp, instruct_gate_mlp = self.instruct_norm1(
            instruct_hidden_states, temb
        )
        instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(instruct_hidden_states, temb)

        # Joint image<->instruction attention, then split back into streams.
        joint_attn_out = self.img_instruct_attn(img_norm1_out, instruct_norm1_out, joint_rotary_emb)
        instruct_attn_out = joint_attn_out[:, :l_instruct]
        img_attn_out = joint_attn_out[:, l_instruct:]

        # Image self-attention.
        img_self_attn_out = self.img_self_attn(img_norm3_out, image_rotary_emb)

        # Image residual updates.
        img_hidden_states = img_hidden_states + mx.tanh(img_gate_msa)[:, None] * self.img_attn_norm(img_attn_out)
        img_hidden_states = img_hidden_states + mx.tanh(img_gate_self)[:, None] * self.img_self_attn_norm(
            img_self_attn_out
        )
        img_mlp_input = (1 + img_scale_mlp[:, None]) * img_norm2_out + img_shift_mlp[:, None]
        img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
        img_hidden_states = img_hidden_states + mx.tanh(img_gate_mlp)[:, None] * self.img_ffn_norm2(img_mlp_out)

        # Instruction residual updates.
        instruct_hidden_states = instruct_hidden_states + mx.tanh(instruct_gate_msa)[:, None] * self.instruct_attn_norm(
            instruct_attn_out
        )
        instruct_mlp_input = (1 + instruct_scale_mlp[:, None]) * instruct_norm2_out + instruct_shift_mlp[:, None]
        instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_mlp_input))
        instruct_hidden_states = instruct_hidden_states + mx.tanh(instruct_gate_mlp)[:, None] * self.instruct_ffn_norm2(
            instruct_mlp_out
        )

        return img_hidden_states, instruct_hidden_states
