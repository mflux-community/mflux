from typing import Callable

import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_attention import Qwen3VLAttention, Qwen3VLKVCache
from mflux.models.common_models.qwen3_vl.qwen3_vl_mlp import Qwen3VLMLP
from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mrope_section: list[int] | None,
        attention_bias: bool,
        rms_norm_eps: float,
        intermediate_size: int,
        linear_class: Callable[..., nn.Module] = nn.Linear,
    ):
        super().__init__()
        self.input_layernorm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3VLAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            linear_class=linear_class,
        )
        self.post_attention_layernorm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3VLMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            linear_class=linear_class,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        past_key_value: Qwen3VLKVCache | None = None,
        max_cache_length: int | None = None,
    ) -> mx.array | tuple[mx.array, Qwen3VLKVCache]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            max_cache_length=max_cache_length,
        )
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, past_key_value
