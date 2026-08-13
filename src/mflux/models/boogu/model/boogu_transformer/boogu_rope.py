from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# Pair of (cos, sin) rotary tables, each shaped (seq_len, sum(axes_dim) // 2).
RotaryEmb = tuple[mx.array, mx.array]


@dataclass
class BooguRotaryEmbeddings:
    """Rotary embeddings for a single forward pass (Turbo T2I, batch size 1).

    Each field is a ``(cos, sin)`` pair of ``float32`` arrays of shape
    ``(seq_len, sum(axes_dim) // 2)`` ready to be applied to query/key tensors
    laid out as ``(B, S, H, head_dim)``.

    Attributes:
        joint: Rotary embedding over the fused ``[instruction, image]`` sequence.
        context: Rotary embedding over the instruction tokens only.
        image: Rotary embedding over the image tokens only.
    """

    joint: RotaryEmb
    context: RotaryEmb
    image: RotaryEmb


class BooguImageRoPE:
    """3D axial RoPE for Boogu-Image (OmniGen2 / Lumina2 lineage).

    Ports ``BooguImageDoubleStreamRotaryPosEmbed`` from the reference. Upstream
    builds complex ``freqs_cis`` via ``get_1d_rotary_pos_embed`` per axis, gathers
    them with 3D ``position_ids``, and applies them as a complex rotation. Here we
    precompute the equivalent ``cos``/``sin`` tables per axis and gather by integer
    position id, which is numerically identical to the complex path.

    The temporal axis (axis 0) is held constant at ``cap_len`` for every image
    token, while axes 1/2 carry the row/column grid coordinates. Instruction
    tokens use the same scalar position on all three axes.

    Args:
        theta: RoPE base frequency (10000 in the reference).
        axes_dim: Per-axis rotary dimension; must sum to ``head_dim``.
        axes_lens: Maximum position table length per axis.
        patch_size: Latent patch size (image token grid is ``H_lat // p`` × ``W_lat // p``).
    """

    def __init__(
        self,
        theta: int = 10000,
        axes_dim: tuple[int, int, int] = (40, 40, 40),
        axes_lens: tuple[int, int, int] = (2048, 1664, 1664),
        patch_size: int = 2,
    ) -> None:
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

        # Per-axis cos/sin tables, each (axes_lens[i], axes_dim[i] // 2).
        self._cos_tables: list[mx.array] = []
        self._sin_tables: list[mx.array] = []
        for dim, length in zip(axes_dim, axes_lens):
            cos_t, sin_t = self._build_axis_table(dim, length, theta)
            self._cos_tables.append(mx.array(cos_t))
            self._sin_tables.append(mx.array(sin_t))

    @staticmethod
    def _build_axis_table(dim: int, length: int, theta: int) -> tuple[np.ndarray, np.ndarray]:
        """Build the (cos, sin) frequency table for one axis (matches get_1d_rotary_pos_embed)."""
        assert dim % 2 == 0, "RoPE axis dim must be even"
        scales = np.arange(0, dim, 2, dtype=np.float64) / dim
        omega = 1.0 / (theta**scales)  # (dim // 2,)
        positions = np.arange(length, dtype=np.float64)
        freqs = np.outer(positions, omega)  # (length, dim // 2)
        return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

    def _gather(self, position_ids: np.ndarray) -> RotaryEmb:
        """Gather per-axis tables at the given 3D position ids and concatenate.

        Args:
            position_ids: Integer array of shape ``(seq_len, 3)``.

        Returns:
            ``(cos, sin)`` arrays of shape ``(seq_len, sum(axes_dim) // 2)``.
        """
        cos_parts, sin_parts = [], []
        for axis in range(3):
            idx = mx.array(position_ids[:, axis].astype(np.int32))
            cos_parts.append(self._cos_tables[axis][idx])
            sin_parts.append(self._sin_tables[axis][idx])
        return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)

    def __call__(self, cap_len: int, h_tokens: int, w_tokens: int) -> BooguRotaryEmbeddings:
        """Build rotary embeddings for a single T2I sample.

        Args:
            cap_len: Number of instruction (caption) tokens.
            h_tokens: Image token rows (``H_lat // patch_size``).
            w_tokens: Image token columns (``W_lat // patch_size``).

        Returns:
            A :class:`BooguRotaryEmbeddings` with joint/context/image rotaries.
        """
        img_len = h_tokens * w_tokens
        seq_len = cap_len + img_len

        position_ids = np.zeros((seq_len, 3), dtype=np.int32)

        # Instruction tokens: identical position on all three axes.
        text_pos = np.arange(cap_len, dtype=np.int32)
        position_ids[:cap_len, 0] = text_pos
        position_ids[:cap_len, 1] = text_pos
        position_ids[:cap_len, 2] = text_pos

        # Image tokens: axis 0 fixed at cap_len, axes 1/2 carry the row/col grid.
        rows = np.repeat(np.arange(h_tokens, dtype=np.int32), w_tokens)
        cols = np.tile(np.arange(w_tokens, dtype=np.int32), h_tokens)
        position_ids[cap_len:, 0] = cap_len
        position_ids[cap_len:, 1] = rows
        position_ids[cap_len:, 2] = cols

        joint_cos, joint_sin = self._gather(position_ids)
        context = (joint_cos[:cap_len], joint_sin[:cap_len])
        image = (joint_cos[cap_len:], joint_sin[cap_len:])
        return BooguRotaryEmbeddings(
            joint=(joint_cos, joint_sin),
            context=context,
            image=image,
        )


def apply_rotary_emb(x: mx.array, rotary_emb: RotaryEmb) -> mx.array:
    """Apply complex rotary embedding to a query/key tensor.

    Mirrors the reference ``apply_rotary_emb(..., use_real=False)``: the last
    dimension is viewed as ``head_dim // 2`` complex pairs ``(real, imag)`` and
    rotated by ``cos + i*sin``.

    Args:
        x: Tensor of shape ``(B, S, H, head_dim)``.
        rotary_emb: ``(cos, sin)`` arrays of shape ``(S, head_dim // 2)``.

    Returns:
        Rotated tensor of shape ``(B, S, H, head_dim)`` in the dtype of ``x``.
    """
    cos, sin = rotary_emb
    x_float = x.astype(mx.float32)
    pairs = mx.reshape(x_float, (*x_float.shape[:-1], x_float.shape[-1] // 2, 2))
    x_real = pairs[..., 0]
    x_imag = pairs[..., 1]

    # cos/sin: (S, D/2) -> (1, S, 1, D/2) to broadcast over batch and heads.
    cos_b = cos[None, :, None, :]
    sin_b = sin[None, :, None, :]

    out_real = x_real * cos_b - x_imag * sin_b
    out_imag = x_real * sin_b + x_imag * cos_b

    out = mx.stack([out_real, out_imag], axis=-1)
    out = mx.reshape(out, x.shape)
    return out.astype(x.dtype)
