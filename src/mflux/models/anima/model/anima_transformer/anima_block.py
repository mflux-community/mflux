import mlx.core as mx
from mlx import nn

from mflux.models.anima.model.anima_transformer.anima_attention import AnimaAttention
from mflux.models.common.compute_precision import ComputePrecision


class AnimaFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=False)
        self.compute_precision = ComputePrecision()

    def __call__(self, x: mx.array) -> mx.array:
        # Measured on the real weights, the activations stay under 1e3, so float16 needs no headroom here.
        precision = self.compute_precision
        dtype = x.dtype
        out = self.layer2(nn.gelu(self.layer1(precision.to_compute(x))))
        return precision.to_stream(out, dtype)


class AnimaBlock(nn.Module):
    # Self-attention, cross-attention to the text, then the MLP. Each step has its own AdaLN shift, scale
    # and gate: a low-rank projection of the timestep embedding plus the shared adaln_lora term.
    def __init__(self, dim: int, context_dim: int, num_heads: int, mlp_ratio: float, adaln_lora_dim: int):
        super().__init__()
        self.self_attn = AnimaAttention(dim, dim, num_heads)
        self.cross_attn = AnimaAttention(dim, context_dim, num_heads)
        self.mlp = AnimaFeedForward(dim, int(dim * mlp_ratio))
        self.adaln_modulation_self_attn = AnimaBlock.modulation(dim, adaln_lora_dim, 3)
        self.adaln_modulation_cross_attn = AnimaBlock.modulation(dim, adaln_lora_dim, 3)
        self.adaln_modulation_mlp = AnimaBlock.modulation(dim, adaln_lora_dim, 3)

    def __call__(
        self,
        x: mx.array,
        emb: mx.array,
        adaln_lora: mx.array,
        context: mx.array,
        rope: tuple[mx.array, mx.array],
    ) -> mx.array:
        shift, scale, gate = AnimaBlock.modulate(self.adaln_modulation_self_attn, emb, adaln_lora)
        h = AnimaBlock.norm(x) * (1 + scale) + shift
        x = x + gate * self.self_attn(h, h, rope)

        shift, scale, gate = AnimaBlock.modulate(self.adaln_modulation_cross_attn, emb, adaln_lora)
        h = AnimaBlock.norm(x) * (1 + scale) + shift
        x = x + gate * self.cross_attn(h, context)

        shift, scale, gate = AnimaBlock.modulate(self.adaln_modulation_mlp, emb, adaln_lora)
        h = AnimaBlock.norm(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)

    @staticmethod
    def modulation(dim: int, adaln_lora_dim: int, chunks: int) -> list[nn.Module]:
        # Indexed like the reference's nn.Sequential, so the checkpoint keys .1.weight and .2.weight map directly.
        return [nn.SiLU(), nn.Linear(dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, chunks * dim, bias=False)]  # fmt: off

    @staticmethod
    def modulate(layers: list[nn.Module], emb: mx.array, adaln_lora: mx.array) -> list[mx.array]:
        out = emb
        for layer in layers:
            out = layer(out)
        out = out + adaln_lora[..., : out.shape[-1]]
        return mx.split(out, out.shape[-1] // emb.shape[-1], axis=-1)

    @staticmethod
    def norm(x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, None, None, 1e-6)
