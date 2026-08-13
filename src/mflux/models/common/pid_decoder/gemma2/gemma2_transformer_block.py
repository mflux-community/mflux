import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.gemma2.gemma2_attention import Gemma2Attention
from mflux.models.common.pid_decoder.gemma2.gemma2_config import Gemma2Config
from mflux.models.common.pid_decoder.gemma2.gemma2_mlp import Gemma2MLP
from mflux.models.common.pid_decoder.gemma2.gemma2_rms_norm import Gemma2RMSNorm


class Gemma2TransformerBlock(nn.Module):
    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.self_attn = Gemma2Attention(config)
        self.mlp = Gemma2MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)
