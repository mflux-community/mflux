import math

import mlx.core as mx
from mlx import nn

from mflux.models.fibo2.model.fibo2_transformer.fibo2_attention import Fibo2Attention
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer_block import Fibo2TransformerBlock
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward


class Fibo2PerceiverLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, cross_attention_dim: int, norm_eps: float):
        super().__init__()
        self.cross_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attention = Fibo2Attention(dim, n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
        self.self_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.self_attention = Fibo2Attention(dim, n_heads, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim=dim, hidden_dim=Fibo2TransformerBlock.ffn_hidden_dim(dim))

    def __call__(
        self,
        latents: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        latents = latents + self.cross_attention(self.cross_attn_norm(latents), encoder_hidden_states=encoder_hidden_states)  # fmt: off
        latents = latents + self.self_attention(self.self_attn_norm(latents))
        return latents + self.ffn(self.ffn_norm(latents))


class Fibo2Perceiver(nn.Module):
    # Squeezes the text encoder's hidden states into a number of gist tokens that grows with the text length
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cross_attention_dim: int,
        num_layers: int,
        norm_eps: float,
        min_gist: int = 256,
        max_gist: int = 512,
        gist_step: int = 64,
        min_text_len: int = 256,
        max_text_len: int = 2048,
    ):
        super().__init__()
        self.min_gist = min_gist
        self.max_gist = max_gist
        self.gist_step = gist_step
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.latent_queries = mx.zeros((max_gist, dim))
        self.layers = [Fibo2PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps) for _ in range(num_layers)]
        self.norm_out = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        encoder_hidden_states: mx.array,
        num_gist_tokens: int,
    ) -> mx.array:
        batch_size = encoder_hidden_states.shape[0]
        latents = mx.broadcast_to(self.latent_queries[:num_gist_tokens][None], (batch_size, num_gist_tokens, self.latent_queries.shape[1]))  # fmt: off
        for layer in self.layers:
            latents = layer(latents, encoder_hidden_states)
        return self.norm_out(latents)

    def gist_count(self, text_length: int) -> int:
        ratio = min(1.0, max(0.0, (float(text_length) - self.min_text_len) / (self.max_text_len - self.min_text_len)))
        raw = self.min_gist + ratio * (self.max_gist - self.min_gist)
        quantized = int(math.ceil(raw / self.gist_step)) * self.gist_step
        return max(self.min_gist, min(self.max_gist, quantized))
