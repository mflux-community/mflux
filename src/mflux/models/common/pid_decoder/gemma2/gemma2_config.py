from dataclasses import dataclass


@dataclass
class Gemma2Config:
    # Defaults verified against google/gemma-2-2b-it/config.json (HF Hub, 2026-07-24).
    vocab_size: int = 256000
    hidden_size: int = 2304
    num_hidden_layers: int = 26
    intermediate_size: int = 9216
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    query_pre_attn_scalar: float = 256.0
