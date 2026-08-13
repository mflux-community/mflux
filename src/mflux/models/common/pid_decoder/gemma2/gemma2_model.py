import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.gemma2.gemma2_config import Gemma2Config
from mflux.models.common.pid_decoder.gemma2.gemma2_rms_norm import Gemma2RMSNorm
from mflux.models.common.pid_decoder.gemma2.gemma2_transformer_block import Gemma2TransformerBlock


class Gemma2Model(nn.Module):
    def __init__(self, config: Gemma2Config = Gemma2Config()):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma2TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        h = self.embed_tokens(input_ids)
        h = h * (self.config.hidden_size**0.5)

        # Gemma-2 is decoder-only: every forward pass (no KV cache here) must be
        # causal, matching mlx-vlm's `create_attention_mask` (always causal for
        # L > 1) and HF's `Gemma2Model.forward` (no bidirectional mode). Additive
        # [1, 1, L, L] mask, -inf above the diagonal.
        L = input_ids.shape[-1]
        causal = mx.triu(mx.full((L, L), mx.finfo(mx.float32).min), k=1)
        mask = causal[None, None, :, :]

        if attention_mask is not None:
            # attention_mask: [B, L] with 1 = keep, 0 = pad. Build an additive
            # [B, 1, 1, L] mask so padded keys get -inf before softmax, then
            # combine with the causal mask (both additive, -inf-style).
            padding = (1.0 - attention_mask[:, None, None, :].astype(mx.float32)) * mx.finfo(mx.float32).min
            mask = mask + padding

        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)
