import mlx.core as mx


class MOSSImageRoPE:
    # Four-axis rotary embedding over (r, f, h, w) coordinates. The sequence is the image
    # tokens followed by the text tokens. The frequency bands are derived constants, so the
    # checkpoint's rope.bands tensor is not loaded.
    def __init__(self, head_dim: int, base: float):
        num_bands = head_dim // 8
        exponent = mx.arange(num_bands, dtype=mx.float32) / num_bands
        self._bands = 1.0 / (base**exponent)

    def __call__(self, height: int, width: int, text_length: int, dtype: mx.Dtype) -> mx.array:
        h = mx.repeat(mx.arange(height, dtype=mx.float32), width) - (height - 1) / 2
        w = mx.tile(mx.arange(width, dtype=mx.float32), height) - (width - 1) / 2
        image = mx.stack([mx.zeros_like(h), mx.full(h.shape, text_length, dtype=mx.float32), h, w], axis=-1)
        text_index = mx.arange(text_length, dtype=mx.float32)
        text_zeros = mx.zeros_like(text_index)
        text = mx.stack([text_zeros, text_index, text_zeros, text_zeros], axis=-1)
        phase = mx.concatenate([image, text], axis=0)[..., None] * self._bands
        rope = mx.concatenate([mx.sin(phase), mx.cos(phase)], axis=-2)
        return rope.reshape(1, phase.shape[0], -1).astype(dtype)

    @staticmethod
    @mx.compile  # fuses the split, concatenate and multiply-add into one kernel
    def rotate(x: mx.array, rope: mx.array) -> mx.array:
        # x: (B, L, heads, head_dim), rope: (1, L, head_dim) as [sin | cos]
        sin, cos = mx.split(rope, 2, axis=-1)
        sin = mx.concatenate([sin, sin], axis=-1)[:, :, None, :]
        cos = mx.concatenate([cos, cos], axis=-1)[:, :, None, :]
        a, b = mx.split(x, 2, axis=-1)
        return x * cos + mx.concatenate([-b, a], axis=-1) * sin
