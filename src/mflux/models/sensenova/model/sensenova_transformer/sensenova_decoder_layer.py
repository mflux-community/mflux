import mlx.core as mx
from mlx import nn

from mflux.models.sensenova.model.sensenova_transformer.sensenova_attention import SenseNovaAttention
from mflux.models.sensenova.model.sensenova_transformer.sensenova_mlp import SenseNovaMLP
from mflux.models.sensenova.model.sensenova_transformer.sensenova_rms_norm import SenseNovaRMSNorm


class SenseNovaDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 12288, eps: float = 1e-6):
        super().__init__()
        self.input_layernorm = SenseNovaRMSNorm(hidden_size, eps)
        self.self_attn = SenseNovaAttention(hidden_size, eps=eps)
        self.post_attention_layernorm = SenseNovaRMSNorm(hidden_size, eps)
        self.mlp = SenseNovaMLP(hidden_size, intermediate_size)

    def __call__(
        self,
        x: mx.array,
        rope: list[tuple[mx.array, mx.array]],
        prefix_kv: tuple[mx.array, mx.array] | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), rope, prefix_kv, mask)
        return x + self.mlp(self.post_attention_layernorm(x))
