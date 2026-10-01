import mlx.core as mx

from mflux.models.qwen21.model.qwen21_text_encoder.language_model import LanguageModel


class Qwen21TextEncoder(LanguageModel):
    # Qwen3-VL text stack of Qwen-Image-2.1: interleaved mrope, but text-only inputs use
    # one shared position ladder replicated across the three mrope axes.

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 12288,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        mrope_section: list[int] | None = None,
    ):
        super().__init__(
            dict(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                head_dim=head_dim,
                attention_bias=False,
                rope_scaling={"mrope_section": mrope_section},
            ),
            text_mode=True,
        )

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        mask_dtype = mx.float32
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = mx.where(
            idx[None, :] > idx[:, None],
            mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
            mx.zeros((seq_len, seq_len), dtype=mask_dtype),
        )
        attention_mask_4d = mx.broadcast_to(causal[None, None, :, :], (batch_size, 1, seq_len, seq_len)) + padding_mask

        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        return super().__call__(hidden_states, position_ids, attention_mask=attention_mask_4d)
