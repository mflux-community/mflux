# Copyright 2026 The Qwen Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
# MLX adaptation; see the model README for the pinned upstream source.
# Adapted from Qwen/Hugging Face's AutoencoderKLQwenImage21 (Apache-2.0).
import mlx.core as mx
from mlx import nn


class ChannelNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        value = x.astype(mx.float32)
        norm = mx.maximum(mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True)), 1e-12)
        return (value / norm).astype(x.dtype) * (x.shape[-1] ** 0.5) * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = ChannelNorm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = ChannelNorm(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv_shortcut(x) + self.conv2(nn.silu(self.norm2(self.conv1(nn.silu(self.norm1(x))))))


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = ChannelNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        q, k, v = mx.split(self.to_qkv(self.norm(x)).reshape(b, 1, h * w, 3 * c), 3, axis=-1)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        return x + self.proj(output.reshape(b, h, w, c))


class MidBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resnets = [ResidualBlock(dim, dim), ResidualBlock(dim, dim)]
        self.attentions = [AttentionBlock(dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Resample(nn.Module):
    def __init__(self, dim: int, up: bool, temporal: bool):
        super().__init__()
        self.up = up
        self.resample = [nn.Identity(), nn.Conv2d(dim, dim, 3, stride=1 if up else 2, padding=1 if up else 0)]
        # The image checkpoint includes these weights; its first-frame path does not use them.
        if temporal:
            self.time_conv = nn.Conv2d(dim, dim * 2 if up else dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        if self.up:
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        else:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.resample[1](x)


class DownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_blocks: int, down: bool, temporal: bool):
        super().__init__()
        self.out_dim = out_dim
        self.factor_s = 2 if down else 1
        self.factor_t = 2 if temporal else 1
        self.resnets = [ResidualBlock(in_dim if i == 0 else out_dim, out_dim) for i in range(num_blocks)]
        self.downsampler = Resample(out_dim, up=False, temporal=temporal) if down else None

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = self._shortcut(x)
        for block in self.resnets:
            x = block(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + shortcut

    def _shortcut(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        s, t = self.factor_s, self.factor_t
        value = x.transpose(0, 3, 1, 2)[:, :, None]
        value = mx.pad(value, [(0, 0), (0, 0), (t - 1, 0), (0, 0), (0, 0)])
        value = value.reshape(b, c, 1, t, h // s, s, w // s, s).transpose(0, 1, 3, 5, 7, 2, 4, 6)
        value = value.reshape(b, self.out_dim, c * t * s * s // self.out_dim, h // s, w // s)
        return mx.mean(value, axis=2).transpose(0, 2, 3, 1)


class UpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_blocks: int, up: bool, temporal: bool):
        super().__init__()
        self.out_dim = out_dim
        self.factor_t = 2 if temporal else 1
        self.resnets = [ResidualBlock(in_dim if i == 0 else out_dim, out_dim) for i in range(num_blocks + 1)]
        self.upsampler = Resample(out_dim, up=True, temporal=temporal) if up else None

    def __call__(self, x: mx.array) -> mx.array:
        original = x
        for block in self.resnets:
            x = block(x)
        if self.upsampler is not None:
            x = self.upsampler(x) + self._shortcut(original)
        return x

    def _shortcut(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        t = self.factor_t
        value = mx.repeat(x.transpose(0, 3, 1, 2), self.out_dim * t * 4 // c, axis=1)
        value = value.reshape(b, self.out_dim, t, 2, 2, 1, h, w).transpose(0, 1, 5, 2, 6, 3, 7, 4)
        value = value.reshape(b, self.out_dim, t, h * 2, w * 2)[:, :, -1]
        return value.transpose(0, 2, 3, 1)
