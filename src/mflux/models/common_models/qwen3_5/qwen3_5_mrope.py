import mlx.core as mx


class Qwen35MRoPE:
    # Interleaved M-RoPE for input that mixes text and image tokens (the Qwen-Image 2.1 small text
    # encoder). Frequency i reads the T, H or W position as [T H W T H W ... T T]. Text-only callers
    # leave it out: Qwen35Attention then uses the 1D mx.fast.rope.
    def __init__(self, head_dim: int, partial_rotary_factor: float, theta: float, mrope_section: list[int]):
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        axis = [0] * (dim // 2)
        for section, offset in ((1, 1), (2, 2)):
            for index in range(offset, mrope_section[section] * 3, 3):
                axis[index] = section
        self.axis = mx.array(axis, dtype=mx.int32)

    def __call__(self, positions: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        # positions: (3, batch, length)
        freqs = positions.astype(mx.float32)[..., None] * self.inv_freq
        freqs = mx.take_along_axis(freqs, self.axis[None, None, None, :], axis=0)[0]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)

    @staticmethod
    def apply(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        rotary = cos.shape[-1]
        x_rot, x_pass = x[..., :rotary], x[..., rotary:]
        half = rotary // 2
        rotated = mx.concatenate([-x_rot[..., half:], x_rot[..., :half]], axis=-1)
        return mx.concatenate([x_rot * cos + rotated * sin, x_pass], axis=-1)
