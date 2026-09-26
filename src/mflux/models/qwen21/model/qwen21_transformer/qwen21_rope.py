from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn


class Qwen21Rope(nn.Module):
    # 3-axis (frame, height, width) rotary embedding over the joint [text, target] sequence.
    # Text tokens advance a shared position on all three axes. The target image freezes the
    # frame axis at the position reached by the text and lays its tokens out on a height/width
    # grid centred on zero, so spatial positions do not depend on the text length.

    def __init__(self, theta: int = 10000, axes_dim: tuple[int, int, int] = (16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = np.arange(8192, dtype=np.int64)
        neg_index = np.arange(1024, dtype=np.int64)[::-1] * -1 - 1
        index = np.concatenate([pos_index, neg_index])
        self.cos_tables = []
        self.sin_tables = []
        for dim in axes_dim:
            freqs = Qwen21Rope._rope_params(index, dim, theta)
            self.cos_tables.append(mx.array(freqs[..., 0]))
            self.sin_tables.append(mx.array(freqs[..., 1]))

    @staticmethod
    def _rope_params(index: np.ndarray, dim: int, theta: int) -> np.ndarray:
        scales = np.arange(0, dim, 2, dtype=np.float32) / dim
        omega = 1.0 / (theta**scales)
        freqs = np.outer(index.astype(np.float32), omega)
        return np.stack([np.cos(freqs), np.sin(freqs)], axis=-1)

    def __call__(self, text_len: int, img_height: int, img_width: int) -> tuple[mx.array, mx.array]:
        frame_index = list(range(text_len))
        position = text_len
        frame_index.extend([position] * (img_height * img_width))

        height_index = list(frame_index)
        width_index = list(frame_index)
        image_height_index = [
            h for h in range(-(img_height - img_height // 2), img_height // 2) for _ in range(img_width)
        ]
        image_width_index = [w for _ in range(img_height) for w in range(-(img_width - img_width // 2), img_width // 2)]
        height_index[text_len:] = image_height_index
        width_index[text_len:] = image_width_index

        return self._to_cos_sin(frame_index, height_index, width_index)

    def __call_edit__(
        self,
        layout: list[tuple],
        target_height: int,
        target_width: int,
    ) -> tuple[mx.array, mx.array]:
        # Edit layout: template-ordered runs -- text runs get a shared 1D ladder on all
        # three axes, image blocks hold the current frame position and consume max(h, w)
        # of budget with centered 2D height/width grids, exactly like the reference
        # QwenImage21Rope block iteration. The target image is the final block.
        frame_index: list[int] = []
        height_index: list[int] = []
        width_index: list[int] = []
        position = 0

        for run in layout:
            if run[0] == "text":
                n = run[1].shape[1]
                frame_index.extend(range(position, position + n))
                height_index.extend(range(position, position + n))
                width_index.extend(range(position, position + n))
                position += n
            else:
                h, w = run[2]
                frame_index.extend([position] * (h * w))
                height_index.extend([hh for hh in range(-(h - h // 2), h // 2) for _ in range(w)])
                width_index.extend([ww for _ in range(h) for ww in range(-(w - w // 2), w // 2)])
                position += max(h, w)

        def _target_block(h: int, w: int) -> None:
            frame_index.extend([position] * (h * w))
            height_index.extend([hh for hh in range(-(h - h // 2), h // 2) for _ in range(w)])
            width_index.extend([ww for _ in range(h) for ww in range(-(w - w // 2), w // 2)])

        _target_block(target_height, target_width)
        return self._to_cos_sin(frame_index, height_index, width_index)

    def _to_cos_sin(
        self, frame_index: list[int], height_index: list[int], width_index: list[int]
    ) -> tuple[mx.array, mx.array]:
        cos = mx.concatenate(
            [
                self.cos_tables[0][mx.array(frame_index)],
                self.cos_tables[1][mx.array(height_index)],
                self.cos_tables[2][mx.array(width_index)],
            ],
            axis=-1,
        )
        sin = mx.concatenate(
            [
                self.sin_tables[0][mx.array(frame_index)],
                self.sin_tables[1][mx.array(height_index)],
                self.sin_tables[2][mx.array(width_index)],
            ],
            axis=-1,
        )
        return cos, sin
