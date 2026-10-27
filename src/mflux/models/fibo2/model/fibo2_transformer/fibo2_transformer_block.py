import mlx.core as mx
from mlx import nn

from mflux.models.fibo2.model.fibo2_transformer.fibo2_attention import Fibo2Attention
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward


class Fibo2TransformerBlock(nn.Module):
    # The Z-Image block. With cross_attention_dim it also reads the text encoder through a tanh-gated cross-attention.
    def __init__(self, dim: int, n_heads: int, norm_eps: float, cross_attention_dim: int | None = None):
        super().__init__()
        self.attention = Fibo2Attention(dim, n_heads, eps=norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=Fibo2TransformerBlock.ffn_hidden_dim(dim))
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]
        self.cross_attention = None
        if cross_attention_dim is not None:
            self.cross_attn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
            self.cross_attention = Fibo2Attention(dim, n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
            self.cross_attn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = mx.zeros((1,))

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        t_noisy: mx.array,
        t_clean: mx.array | None = None,
        noise_mask: mx.array | None = None,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        scale_msa, gate_msa, scale_mlp, gate_mlp = self._modulation(t_noisy, t_clean, noise_mask)

        attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attention_norm2(attn_out)

        if self.cross_attention is not None:
            cross_out = self.cross_attention(self.cross_attn_norm1(x), encoder_hidden_states=encoder_hidden_states)
            x = x + mx.tanh(self.cross_attn_gate) * self.cross_attn_norm2(cross_out)

        ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        return x + gate_mlp * self.ffn_norm2(ffn_out)

    @staticmethod
    def ffn_hidden_dim(dim: int) -> int:
        return (dim * 8 // 3) // 64 * 64

    def _modulation(
        self, t_noisy: mx.array, t_clean: mx.array | None, noise_mask: mx.array | None
    ) -> tuple[mx.array, ...]:
        mod = self.adaLN_modulation[0](t_noisy)[:, None, :]
        if noise_mask is not None:
            # Noisy tokens take the timestep's modulation, clean tokens (gist, context images) the t = 0 one
            mod_clean = self.adaLN_modulation[0](t_clean)[:, None, :]
            mod = mx.where(noise_mask[:, :, None], mod, mod_clean)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(mod, 4, axis=2)
        return 1.0 + scale_msa, mx.tanh(gate_msa), 1.0 + scale_mlp, mx.tanh(gate_mlp)
