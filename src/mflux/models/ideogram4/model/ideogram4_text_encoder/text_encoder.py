from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.ideogram4.model.ideogram4_text_encoder.decoder_layer import Qwen3VLDecoderLayer


class Ideogram4TextEncoder(Qwen3TextEncoder):
    # Ideogram 4 runs the shared Qwen3TextEncoder (the Qwen3-VL-8B text tower) with fp8 decoder layers.
    # The prompt features are the outputs of layers 0, 3, ..., 33 and 35, in hidden-state indices
    # (0 is the embedding output), with the layers interleaved per hidden unit.
    HIDDEN_STATE_LAYERS = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36)

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
        super().__init__(
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
            decoder_layer_class=Qwen3VLDecoderLayer,
        )
