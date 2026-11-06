import mlx.core as mx
from mlx import nn


class Qwen35RMSNorm(nn.Module):
    # Qwen3.5 stores the norm weight as an offset from one: the scale is (1 + weight).
    # fused=True uses mx.fast.rms_norm (the Qwen-Image 2.1 small text encoder); the default
    # composes the ops (MOSS-Image). Both are float32 inside, but the bits are not the same.
    def __init__(self, dims: int, eps: float = 1e-6, fused: bool = False):
        super().__init__()
        self.eps = eps
        self.fused = fused
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        scale = 1.0 + self.weight.astype(mx.float32)
        if self.fused:
            return mx.fast.rms_norm(x.astype(mx.float32), scale, self.eps).astype(x.dtype)
        normed = Qwen35RMSNorm.normalize(x, self.eps)
        return (normed * scale).astype(x.dtype)

    @staticmethod
    def normalize(x: mx.array, eps: float) -> mx.array:
        x = x.astype(mx.float32)
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


class Qwen35GatedRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6, fused: bool = False):
        super().__init__()
        self.eps = eps
        self.fused = fused
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        if self.fused:
            normed = mx.fast.rms_norm(x.astype(mx.float32), None, self.eps).astype(x.dtype)
        else:
            normed = Qwen35RMSNorm.normalize(x, self.eps).astype(x.dtype)
        gated = (self.weight * normed) * nn.silu(gate.astype(mx.float32))
        return gated.astype(x.dtype)
