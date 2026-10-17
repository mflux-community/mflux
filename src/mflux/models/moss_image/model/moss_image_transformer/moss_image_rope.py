import mlx.core as mx
import mlx.nn as nn


def rotate_half(x: mx.array, rope: mx.array) -> mx.array:
    # x: (B, seq_len, heads, head_dim)
    # rope: (1, seq_len, head_dim)
    sin, cos = mx.split(rope, 2, axis=-1)
    sin = mx.expand_dims(mx.concatenate([sin, sin], axis=-1), axis=-2)
    cos = mx.expand_dims(mx.concatenate([cos, cos], axis=-1), axis=-2)
    a, b = mx.split(x, 2, axis=-1)
    return x * cos + mx.concatenate([-b, a], axis=-1) * sin


class MOSSImageRoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.num_bands = head_dim // 8
        self.base = base
        exponent = mx.arange(self.num_bands, dtype=mx.float32) / self.num_bands
        self.bands = 1.0 / (self.base ** exponent)

    def __call__(self, height: int, width: int, text_length: int) -> mx.array:
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h_grid = mx.repeat(mx.arange(height, dtype=mx.float32)[:, None], width, axis=1)
        w_grid = mx.repeat(mx.arange(width, dtype=mx.float32)[None, :], height, axis=0)

        h_flat = mx.reshape(h_grid, (-1,))
        w_flat = mx.reshape(w_grid, (-1,))
        zeros = mx.zeros_like(h_flat)
        text_lens = mx.full_like(h_flat, text_length)

        h_centered = h_flat - (height - 1) / 2.0
        w_centered = w_flat - (width - 1) / 2.0

        image_coords = mx.stack([zeros, text_lens, h_centered, w_centered], axis=-1)

        text_zeros = mx.zeros((text_length, 4), dtype=mx.float32)
        text_idx = mx.arange(text_length, dtype=mx.float32)
        text_coords = mx.concatenate([
            text_zeros[:, :1],
            text_idx[:, None],
            text_zeros[:, 2:]
        ], axis=-1)

        coords = mx.concatenate([image_coords, text_coords], axis=0) # (seq_len, 4)
        phase = mx.expand_dims(coords, axis=-1) * self.bands # (seq_len, 4, num_bands)

        sin_part = mx.sin(phase)
        cos_part = mx.cos(phase)
        rope = mx.concatenate([sin_part, cos_part], axis=-2) # (seq_len, 8, num_bands)
        rope = mx.reshape(rope, (1, height * width + text_length, self.head_dim))
        return rope
