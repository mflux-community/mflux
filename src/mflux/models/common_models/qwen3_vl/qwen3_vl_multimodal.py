import mlx.core as mx
import numpy as np
from mlx import nn


class Qwen3VLMultimodal:
    # What Qwen3-VL needs around its language model and vision tower to run as transformers runs it: the 3D (M-RoPE)
    # positions of a prompt with images, the vision tower's activations and the layout of its patch embedding

    @staticmethod
    def mrope_positions(input_ids: mx.array, grids: mx.array | None, image_token_id: int, merge_size: int) -> mx.array:
        ids = np.asarray(input_ids[0])
        result = np.zeros((3, len(ids)), dtype=np.int32)
        cursor, position = 0, 0
        if grids is not None:
            for t, h, w in grids.tolist():
                h, w = h // merge_size, w // merge_size
                start = int(np.flatnonzero(ids[cursor:] == image_token_id)[0]) + cursor
                result[:, cursor:start] = np.arange(position, position + start - cursor)[None]
                position += start - cursor
                length = t * h * w
                result[:, start : start + length] = (
                    np.stack(
                        [
                            np.repeat(np.arange(t), h * w),
                            np.tile(np.repeat(np.arange(h), w), t),
                            np.tile(np.arange(w), t * h),
                        ]
                    )
                    + position
                )
                cursor = start + length
                position += max(h, w)
        result[:, cursor:] = np.arange(position, position + len(ids) - cursor)[None]
        return mx.array(result[:, None])

    @staticmethod
    def exact_gelu(value: mx.array) -> mx.array:
        return nn.gelu(value.astype(mx.float32)).astype(value.dtype)

    @staticmethod
    def tanh_gelu(value: mx.array) -> mx.array:
        return nn.gelu_approx(value.astype(mx.float32)).astype(value.dtype)

    @staticmethod
    def use_reference_activations(visual) -> None:
        # The vision blocks use the tanh GELU and the mergers the exact one, both in float32
        for block in visual.blocks:
            block.mlp.act_fn = Qwen3VLMultimodal.tanh_gelu
        for merger in [visual.merger, *visual.deepstack_merger_list]:
            merger.act_fn = Qwen3VLMultimodal.exact_gelu

    @staticmethod
    def vision_weight(key: str, weight: mx.array) -> mx.array:
        # The patch embedding is a 3D convolution: PyTorch stores it (out, in, t, h, w), MLX (out, t, h, w, in)
        if key.endswith("patch_embed.proj.weight"):
            return weight.transpose(0, 2, 3, 4, 1)
        return weight
