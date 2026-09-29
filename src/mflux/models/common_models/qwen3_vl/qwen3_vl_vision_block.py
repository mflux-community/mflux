import mlx.core as mx
from mlx import nn

from .qwen3_vl_vision_attention import Qwen3VLVisionAttention
from .qwen3_vl_vision_mlp import Qwen3VLVisionMLP


class Qwen3VLVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu_pytorch_tanh",
        preserve_input_dtype: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(
            hidden_size=hidden_size, num_heads=num_heads, preserve_input_dtype=preserve_input_dtype
        )
        self.mlp = Qwen3VLVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
