import inspect
from collections.abc import Iterator

import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder_layer import Qwen3VLDecoderLayer
from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.common_models.qwen3_vl.qwen3_vl_rope import Qwen3VLRotaryEmbedding


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, hidden: mx.array) -> mx.array:
        value = hidden.astype(mx.float32)
        value *= mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
        # Qwen3-VL rounds the normalized values before multiplying the learned weight.
        return value.astype(hidden.dtype) * self.weight


class Qwen3VLLanguageModel(nn.Module):
    def __init__(self, config: dict, *, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        params = {
            key: value for key, value in config.items() if key in inspect.signature(Qwen3VLDecoderLayer).parameters
        }
        params["mrope_section"] = config["rope_scaling"]["mrope_section"]
        self.layers = [Qwen3VLDecoderLayer(**params) for _ in range(config["num_hidden_layers"])]
        if text_mode:
            self.norm = Qwen3VLRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        else:
            # Only greedy decoding (grounding, prompt rewriting, verification) reads the final norm.
            self.norm = Qwen3VLTextRMSNorm(config["hidden_size"], config["rms_norm_eps"])
            for layer in self.layers:
                layer.input_layernorm = Qwen3VLTextRMSNorm(config["hidden_size"], config["rms_norm_eps"])
                layer.post_attention_layernorm = Qwen3VLTextRMSNorm(config["hidden_size"], config["rms_norm_eps"])
                layer.self_attn.q_norm = Qwen3VLTextRMSNorm(config["head_dim"], config["rms_norm_eps"])
                layer.self_attn.k_norm = Qwen3VLTextRMSNorm(config["head_dim"], config["rms_norm_eps"])
        rope_args = {"max_position_embeddings": config["max_position_embeddings"]} if text_mode else {}
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            dim=config["head_dim"], base=config["rope_theta"], mrope_section=params["mrope_section"], **rope_args
        )

    def __call__(
        self,
        hidden: mx.array,
        positions: mx.array,
        image_indices: mx.array | None = None,
        deepstack: list[mx.array] | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        for hidden in self._decoder_outputs(hidden, positions, image_indices, deepstack, attention_mask):
            if not self.text_mode:
                mx.eval(hidden)
        # Without text_mode the last decoder output is returned before the final RMSNorm, as Qwen-Image-2.1 reads it
        return self.norm(hidden) if self.text_mode else hidden

    def hidden_states(
        self,
        hidden: mx.array,
        positions: mx.array,
        image_indices: mx.array | None = None,
        deepstack: list[mx.array] | None = None,
    ) -> list[mx.array]:
        # transformers' output_hidden_states: the embeddings, then each layer's output, deepstack added
        return [hidden, *self._decoder_outputs(hidden, positions, image_indices, deepstack)]

    def _decoder_outputs(
        self,
        hidden: mx.array,
        positions: mx.array,
        image_indices: mx.array | None,
        deepstack: list[mx.array] | None,
        attention_mask: mx.array | None = None,
    ) -> Iterator[mx.array]:
        rope = self.rotary_emb(hidden, positions)
        if attention_mask is None:
            idx = mx.arange(hidden.shape[1])
            attention_mask = (idx[:, None] >= idx[None, :])[None, None]
        for index, layer in enumerate(self.layers):
            hidden, _ = layer(hidden, attention_mask=attention_mask, position_embeddings=rope)
            if deepstack is not None and index < len(deepstack):
                hidden[:, image_indices] += deepstack[index].astype(hidden.dtype)[None]
            yield hidden
