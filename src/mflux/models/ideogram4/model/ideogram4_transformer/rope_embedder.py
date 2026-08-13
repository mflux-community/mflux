import mlx.core as mx
from mlx import nn


class Ideogram4MRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim != 3 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must have shape (batch, seq, 3)")
        pos = position_ids.astype(mx.float32)
        freqs = []
        for axis in range(3):
            axis_pos = pos[:, :, axis][..., None]
            freqs.append(axis_pos * self.inv_freq[None, None, :])

        axis_selector = [0] * self.inv_freq.shape[0]
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            for idx in range(offset, length, 3):
                axis_selector[idx] = axis
        selector = mx.array(axis_selector, dtype=mx.int32)
        selector = mx.broadcast_to(
            selector[None, None, None, :],
            (position_ids.shape[0], position_ids.shape[1], 1, selector.shape[0]),
        )
        freq_stack = mx.stack(freqs, axis=-2)
        freqs_t = mx.squeeze(mx.take_along_axis(freq_stack, selector, axis=-2), axis=-2)

        emb = mx.concatenate([freqs_t, freqs_t], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    @staticmethod
    def apply_rotary_pos_emb(
        q: mx.array,
        k: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> tuple[mx.array, mx.array]:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q_embed = (q * cos) + (Ideogram4MRoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (Ideogram4MRoPE.rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return mx.concatenate([-x2, x1], axis=-1)
