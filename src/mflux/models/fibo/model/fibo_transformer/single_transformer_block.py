import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_transformer.fibo_single_attention import FiboSingleAttention
from mflux.models.flux.model.flux_transformer.ada_layer_norm_zero_single import AdaLayerNormZeroSingle


class FiboSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        layer: int,
        dim: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.layer = layer
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = AdaLayerNormZeroSingle(dim=dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.gelu_approx
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        self.attn = FiboSingleAttention(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim)  # fmt: off

    def __call__(
        self,
        temb: mx.array,
        hidden_states: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # 0. Residual connection
        residual = hidden_states

        # 1. AdaLayerNormZeroSingle
        norm_hidden_states, gate = self.norm(
            hidden_states=hidden_states,
            text_embeddings=temb,
        )

        # 2. Attention
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        # 3. MLP + projection
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        hidden_states = mx.concatenate([attn_output, mlp_hidden_states], axis=2)
        gate = mx.expand_dims(gate, axis=1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
