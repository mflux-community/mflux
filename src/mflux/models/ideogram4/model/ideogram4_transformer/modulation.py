import mlx.core as mx
from mlx import nn

from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return (self.weight.astype(mx.float32) * x).astype(input_dtype)


class Ideogram4LayerNormNoAffine(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
        return ((x - mean) * mx.rsqrt(variance + self.eps)).astype(input_dtype)


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        self.mlp_in = Fp8Linear(dim, dim, bias=True)
        self.mlp_out = Fp8Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        from mflux.models.common.config import ModelConfig

        x = x.astype(mx.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = self.sinusoidal_embedding(scaled, self.dim).astype(ModelConfig.precision)
        emb = nn.silu(self.mlp_in(emb))
        return self.mlp_out(emb)

    @staticmethod
    def sinusoidal_embedding(t: mx.array, dim: int, scale: float = 1e4) -> mx.array:
        import math

        half = dim // 2
        freq = math.log(scale) / (half - 1)
        freq = mx.exp(mx.arange(half, dtype=mx.float32) * -freq)
        emb = t[..., None] * freq
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        if dim % 2:
            emb = mx.pad(emb, [(0, 0)] * (emb.ndim - 1) + [(0, 1)])
        return emb


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.norm_final = Ideogram4LayerNormNoAffine(hidden_size, eps=1e-6)
        self.linear = Fp8Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = Fp8Linear(adanln_dim, hidden_size, bias=True)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaln_modulation(nn.silu(c))
        return self.linear(self.norm_final(x) * scale)
