# Copyright 2026 The Qwen Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AutoencoderKLQwenImage21; see ../../reference/README.md.
import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_vae.qwen21_avg_down import Qwen21AvgDown
from mflux.models.qwen21.model.qwen21_vae.qwen21_dup_up import Qwen21DupUp


class VAEOperations:
    @staticmethod
    def conv(layer: nn.Module, x: mx.array, text_mode: bool) -> mx.array:
        if isinstance(layer, nn.Identity):
            return x
        if text_mode:
            return layer(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        return layer(x)


class ChannelNorm(nn.Module):
    def __init__(self, channels: int, text_mode: bool = False):
        super().__init__()
        self.gamma = mx.ones((channels,))
        self.text_mode = text_mode
        self.scale = float(channels) ** 0.5

    def __call__(self, x: mx.array) -> mx.array:
        axis = 1 if self.text_mode else -1
        value = x.astype(mx.float32)
        norm = mx.maximum(mx.sqrt(mx.sum(value * value, axis=axis, keepdims=True)), 1e-12)
        value = value / norm
        if self.text_mode:
            return (value * self.scale * self.gamma[None, :, None, None]).astype(x.dtype)
        return value.astype(x.dtype) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        self.norm1 = ChannelNorm(in_dim, text_mode)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = ChannelNorm(out_dim, text_mode)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = VAEOperations.conv(self.conv_shortcut, x, self.text_mode)
        x = VAEOperations.conv(self.conv1, nn.silu(self.norm1(x)), self.text_mode)
        x = VAEOperations.conv(self.conv2, nn.silu(self.norm2(x)), self.text_mode)
        return x + shortcut if self.text_mode else shortcut + x


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        self.norm = ChannelNorm(dim, text_mode)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        normalized = self.norm(x)
        if self.text_mode:
            normalized = normalized.transpose(0, 2, 3, 1)
        b, h, w, c = normalized.shape
        qkv = self.to_qkv(normalized)
        # Keep the text path's explicit attention arithmetic and the edit path's SDPA.
        if self.text_mode:
            qkv = qkv.reshape(b, h * w, 3, c)
            q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]
            scale = 1.0 / mx.sqrt(mx.array(float(c)))
            scores = mx.matmul(q, k.transpose(0, 2, 1)) * scale
            output = mx.matmul(mx.softmax(scores, axis=-1), v)
        else:
            q, k, v = mx.split(qkv.reshape(b, 1, h * w, 3 * c), 3, axis=-1)
            output = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        output = self.proj(output.reshape(b, h, w, c))
        return output.transpose(0, 3, 1, 2) + x if self.text_mode else x + output


class MidBlock(nn.Module):
    def __init__(self, dim: int, text_mode: bool = False):
        super().__init__()
        self.resnets = [ResidualBlock(dim, dim, text_mode), ResidualBlock(dim, dim, text_mode)]
        self.attentions = [AttentionBlock(dim, text_mode)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Resample(nn.Module):
    def __init__(self, dim: int, up: bool, temporal: bool, text_mode: bool = False, out_dim: int | None = None):
        super().__init__()
        self.up = up
        self.text_mode = text_mode
        self.resample = [
            nn.Identity(),
            nn.Conv2d(dim, out_dim or dim, 3, stride=1 if up else 2, padding=1 if up else 0),
        ]
        # Only edit checkpoints retain unused first-frame temporal convolution weights.
        if temporal and not text_mode:
            self.time_conv = nn.Conv2d(dim, dim * 2 if up else dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        if self.text_mode:
            x = x.transpose(0, 2, 3, 1)
        if self.up:
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        else:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.resample[1](x)
        return x.transpose(0, 3, 1, 2) if self.text_mode else x


class DownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_blocks: int, down: bool, temporal: bool, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        self.resnets = [ResidualBlock(in_dim if i == 0 else out_dim, out_dim, text_mode) for i in range(num_blocks)]
        self.downsampler = Resample(out_dim, up=False, temporal=temporal, text_mode=text_mode) if down else None
        self.avg_shortcut = Qwen21AvgDown(in_dim, out_dim, 2 if temporal else 1, 2 if down else 1)

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = self._shortcut(x)
        for block in self.resnets:
            x = block(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + shortcut

    def _shortcut(self, x: mx.array) -> mx.array:
        shortcut = self.avg_shortcut(x if self.text_mode else x.transpose(0, 3, 1, 2))
        return shortcut if self.text_mode else shortcut.transpose(0, 2, 3, 1)


class UpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_blocks: int, up: bool, temporal: bool, text_mode: bool = False):
        super().__init__()
        self.text_mode = text_mode
        self.resnets = [ResidualBlock(in_dim if i == 0 else out_dim, out_dim, text_mode) for i in range(num_blocks + 1)]
        self.upsampler = Resample(out_dim, up=True, temporal=temporal, text_mode=text_mode) if up else None
        self.avg_shortcut = Qwen21DupUp(in_dim, out_dim, 2 if temporal else 1) if up else None

    def __call__(self, x: mx.array) -> mx.array:
        original = x
        for block in self.resnets:
            x = block(x)
        if self.upsampler is not None:
            shortcut = self.avg_shortcut(original if self.text_mode else original.transpose(0, 3, 1, 2))
            if not self.text_mode:
                shortcut = shortcut.transpose(0, 2, 3, 1)
            x = self.upsampler(x) + shortcut
        return x
