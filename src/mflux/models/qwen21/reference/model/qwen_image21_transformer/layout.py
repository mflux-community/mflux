# Copyright 2026 The Qwen Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
# MLX adaptation; see the model README for the pinned upstream source.
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class QwenImage21Layout:
    repeat_indices: mx.array
    image_indices: mx.array
    text_indices: mx.array
    target_mask: mx.array
    rope: tuple[mx.array, mx.array]
    segments: list[tuple[int, int, bool]]
    target_tokens: int
    prefix_length: int

    @staticmethod
    def create(image_slots: mx.array, shapes: list[tuple[int, int, int]], axes: tuple[int, ...]) -> "QwenImage21Layout":
        slots = np.asarray(image_slots, dtype=bool).reshape(-1)
        target_tokens = int(np.prod(shapes[-1]))
        if target_tokens % 4:
            raise ValueError("The target must contain a multiple of four latent tokens.")
        slots = np.concatenate([slots, np.ones(target_tokens // 4, dtype=bool)])
        repeats = np.where(slots, 4, 1)
        indices = np.repeat(np.arange(len(slots)), repeats)
        image_mask = slots[indices]
        image_indices = np.flatnonzero(image_mask)
        lengths = [int(np.prod(shape)) for shape in shapes]
        if sum(lengths) != len(image_indices):
            raise ValueError("Reference image slots do not match the VAE latent shapes.")
        image_ids = np.full(len(indices), -1)
        image_ids[image_indices] = np.repeat(np.arange(len(shapes)), lengths)
        prefix = len(indices) - target_tokens
        target_mask = np.arange(len(indices)) >= prefix
        frame = np.zeros(len(indices), dtype=np.float32)
        height = frame.copy()
        width = frame.copy()
        cursor, position, offset = 0, 0, 0
        for (_, h, w), length in zip(shapes, lengths):
            start = int(image_indices[offset])
            if not np.all(image_mask[start : start + length]):
                raise ValueError("Reference image tokens must form contiguous blocks.")
            for axis in (frame, height, width):
                axis[cursor:start] = np.arange(position, position + start - cursor)
            position += start - cursor
            frame[start : start + length] = position
            height[start : start + length] = np.repeat(np.arange(-(h - h // 2), h // 2), w)
            width[start : start + length] = np.tile(np.arange(-(w - w // 2), w // 2), h)
            position += max(h, w)
            cursor = start + length
            offset += length
        angles = np.concatenate(
            [
                axis[:, None] * (10000.0 ** (-np.arange(0, dim, 2, dtype=np.float32) / dim))[None]
                for axis, dim in zip((frame, height, width), axes)
            ],
            axis=-1,
        )
        segments = []
        start = 0
        for end in range(1, prefix + 1):
            if end == prefix or image_ids[end] != image_ids[start]:
                segments.append((start, end, bool(image_ids[start] < 0)))
                start = end
        return QwenImage21Layout(
            repeat_indices=mx.array(indices, dtype=mx.int32),
            image_indices=mx.array(image_indices, dtype=mx.int32),
            text_indices=mx.array(np.flatnonzero(~image_mask), dtype=mx.int32),
            target_mask=mx.array(target_mask),
            rope=(mx.array(np.cos(angles)), mx.array(np.sin(angles))),
            segments=segments,
            target_tokens=target_tokens,
            prefix_length=prefix,
        )

    @staticmethod
    def rotate(x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
        cos, sin = (value[None, None] for value in rope)
        pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs[..., 0], pairs[..., 1]
        return mx.stack([real * cos - imag * sin, imag * cos + real * sin], axis=-1).reshape(x.shape).astype(x.dtype)
