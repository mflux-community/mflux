# Copyright 2026 The Qwen Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
# MLX adaptation; see the model README for the pinned upstream source.
# Adapted from Qwen/Hugging Face's QwenImage21Transformer2DModel (Apache-2.0).
import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout


class ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        value = x.astype(mx.float32)
        value *= mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
        return (value * (self.weight.astype(mx.float32) + 1)).astype(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, context_dim: int, dim: int, eps: float):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_dim, eps)
        self.in_layer = nn.Linear(context_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.gelu_approx(self.in_layer(self.text_norm(x))))


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.timestep_embedder = TimestepMLP(dim)

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> mx.array:
        frequency = mx.exp(-mx.log(mx.array(10000.0)) * mx.arange(128, dtype=mx.float32) / 128)
        phase = timestep.astype(mx.float32)[:, None] * 1000 * frequency[None]
        return self.timestep_embedder(mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1).astype(dtype))


class TimestepMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim, bias=False)
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * ratio, bias=False)
        self.gate_layer = nn.Linear(dim, dim * ratio, bias=False)
        self.out = nn.Linear(dim * ratio, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out(nn.silu(self.gate_layer(x)) * self.proj(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)

    def __call__(
        self,
        x: mx.array,
        layout: QwenImage21Layout,
        rope: tuple[mx.array, mx.array],
        cached: tuple[mx.array, mx.array] | None = None,
        extract: bool = False,
        key_valid: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        b, length, dim = x.shape
        q, k, v = [
            layer(x).reshape(b, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            for layer in (self.to_q, self.to_k, self.to_v)
        ]
        q = QwenImage21Layout.rotate(self.norm_q(q).astype(v.dtype), rope)
        k = QwenImage21Layout.rotate(self.norm_k(k).astype(v.dtype), rope)
        stored = None
        if cached is not None:
            k, v = mx.concatenate([cached[0], k], axis=2), mx.concatenate([cached[1], v], axis=2)
        elif extract:
            stored = (mx.contiguous(k[:, :, : layout.prefix_length]), mx.contiguous(v[:, :, : layout.prefix_length]))
        scale = self.head_dim**-0.5
        if cached is not None:
            mask = key_valid[:, None, None, :] if key_valid is not None else None
            output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        else:
            outputs = []
            for start, end, is_text in [*layout.segments, (layout.prefix_length, length, False)]:
                mask = None
                if is_text:
                    mask = mx.arange(end)[None, :] <= mx.arange(start, end)[:, None]
                if key_valid is not None:
                    valid = key_valid[:, None, None, :end]
                    mask = valid if mask is None else mask[None, None] & valid
                outputs.append(
                    mx.fast.scaled_dot_product_attention(
                        q[:, :, start:end], k[:, :, :end], v[:, :, :end], scale=scale, mask=mask
                    )
                )
            output = mx.concatenate(outputs, axis=2)
        return self.to_out[0](output.transpose(0, 2, 1, 3).reshape(b, length, dim)), stored


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ratio: int, eps: float):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn = Attention(dim, heads, head_dim, eps)
        self.img_mlp = FeedForward(dim, ratio)

    def __call__(
        self,
        x: mx.array,
        modulation: mx.array,
        target_mask: mx.array,
        layout: QwenImage21Layout,
        rope: tuple[mx.array, mx.array],
        cached: tuple[mx.array, mx.array] | None = None,
        extract: bool = False,
        key_valid: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        scale1, gate1, scale2, gate2 = [self.select_rows(v, target_mask) for v in mx.split(modulation, 4, axis=-1)]
        attn, stored = self.attn(self.img_norm1(x) * (1 + scale1), layout, rope, cached, extract, key_valid)
        x = x + mx.tanh(gate1) * attn
        x = x + mx.tanh(gate2) * self.img_mlp(self.img_norm2(x) * (1 + scale2))
        return x, stored

    @staticmethod
    def select_rows(value: mx.array, mask: mx.array) -> mx.array:
        return mx.where(mask[None, :, None], value[:-1, None], value[-1:, None])


class FinalNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)

    def __call__(self, x: mx.array, time: mx.array, mask: mx.array) -> mx.array:
        return self.norm(x) * (1 + TransformerBlock.select_rows(self.linear(nn.silu(time)), mask))
