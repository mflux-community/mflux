import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.sensenova.model.sensenova_transformer.sensenova_decoder_layer import SenseNovaDecoderLayer
from mflux.models.sensenova.model.sensenova_transformer.sensenova_rope import SenseNovaRope


class SenseNovaTextEncoder(nn.Module):
    # The und stream of the MoT backbone. It only produces the per-layer prefix KV the gen stream attends to.
    def __init__(
        self,
        num_layers: int = 42,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [SenseNovaDecoderLayer(hidden_size, intermediate_size, eps) for _ in range(num_layers)]

    def __call__(self, input_ids: mx.array) -> list[tuple[mx.array, mx.array]]:
        # input_ids: [B, L]; returns per layer (k, v) [B, 8, L, 128], k post-norm and post-RoPE
        hidden = self.embed_tokens(input_ids)
        rope = SenseNovaRope.cos_sin(SenseNovaRope.text_indexes(input_ids.shape[1]), hidden.dtype)
        kv = []
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            q, k, v = attn.project(layer.input_layernorm(hidden), rope)
            kv.append((k, v))
            if i == len(self.layers) - 1:
                break
            out = scaled_dot_product_attention(q, k, v, scale=attn.scale, mask="causal")
            hidden = hidden + attn.o_proj(out.transpose(0, 2, 1, 3).reshape(hidden.shape))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return kv
