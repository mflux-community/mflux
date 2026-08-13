import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_rotary_embedding import Qwen3TextRotaryEmbedding
from mflux.models.ideogram4.model.ideogram4_text_encoder.decoder_layer import Qwen3VLDecoderLayer


class Qwen3TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 12288,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3TextRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        tap_layers: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35),
    ) -> list[mx.array]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        if position_ids is None:
            position_ids = mx.broadcast_to(
                mx.arange(seq_len, dtype=mx.int32)[None, :],
                (batch_size, seq_len),
            )

        mask_dtype = hidden_states.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = padding_mask[:, None, None, :]

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = idx[None, :] > idx[:, None]
        causal_mask = mx.where(
            causal,
            mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
            mx.zeros((seq_len, seq_len), dtype=mask_dtype),
        )
        causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_mask + padding_mask
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        tap_set = set(tap_layers)
        captured: dict[int, mx.array] = {}
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states
        return [captured[i] for i in tap_layers]

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        position_ids: mx.array,
        tap_layers: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35),
    ) -> mx.array:
        selected = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            tap_layers=tap_layers,
        )
        stacked = mx.stack(selected, axis=0)
        stacked = mx.transpose(stacked, (1, 2, 3, 0))
        batch_size, seq_len, hidden_dim, num_layers = stacked.shape
        return stacked.reshape(batch_size, seq_len, hidden_dim * num_layers)
