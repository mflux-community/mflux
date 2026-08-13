from __future__ import annotations

import mlx.core as mx

from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder

# Boogu-Image-0.1 instruction encoder = Qwen3-VL text decoder (mllm/config.json).
BOOGU_TEXT_ENCODER_CONFIG = dict(
    vocab_size=151936,
    hidden_size=4096,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=12288,
    max_position_embeddings=262144,
    rope_theta=5000000.0,
    rms_norm_eps=1e-6,
    head_dim=128,
)


class BooguTextEncoder(Qwen3TextEncoder):
    """Qwen3-VL text decoder used as Boogu's instruction encoder.

    Reuses Flux2's parameterized ``Qwen3TextEncoder`` (verified to match Boogu's
    ``model.language_model.*`` checkpoint keys exactly) with Boogu's config. For
    text-only T2I the M-RoPE collapses to standard 1D RoPE, so the 1D rotary path
    is exact.
    """

    def __init__(self) -> None:
        super().__init__(**BOOGU_TEXT_ENCODER_CONFIG)

    def get_instruction_features(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """Return the final-layer (normed) hidden states = the ``last_hidden_state``.

        Matches the reference ``mllm.model(...).last_hidden_state`` used to build
        the transformer's ``instruction_hidden_states`` (reduce_type=mean over a
        single layer is a no-op).

        Args:
            input_ids: ``(B, L)`` token ids from the Qwen3-VL chat template.
            attention_mask: Optional ``(B, L)`` mask.

        Returns:
            ``(B, L, hidden_size)`` instruction features.
        """
        hidden_states, _ = self(input_ids, attention_mask=attention_mask, output_hidden_states=False)
        return hidden_states
