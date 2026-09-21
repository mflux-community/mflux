import inspect

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder_layer import Qwen3VLDecoderLayer
from mflux.models.common_models.qwen3_vl.qwen3_vl_rope import Qwen3VLRotaryEmbedding
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel


class TextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, hidden: mx.array) -> mx.array:
        value = hidden.astype(mx.float32)
        value *= mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
        # Qwen3-VL rounds the normalized values before multiplying the learned weight.
        return value.astype(hidden.dtype) * self.weight


class LanguageModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        params = {
            key: value for key, value in config.items() if key in inspect.signature(Qwen3VLDecoderLayer).parameters
        }
        params["mrope_section"] = config["rope_scaling"]["mrope_section"]
        self.layers = [Qwen3VLDecoderLayer(**params) for _ in range(config["num_hidden_layers"])]
        for layer in self.layers:
            layer.input_layernorm = TextRMSNorm(config["hidden_size"], config["rms_norm_eps"])
            layer.post_attention_layernorm = TextRMSNorm(config["hidden_size"], config["rms_norm_eps"])
            layer.self_attn.q_norm = TextRMSNorm(config["head_dim"], config["rms_norm_eps"])
            layer.self_attn.k_norm = TextRMSNorm(config["head_dim"], config["rms_norm_eps"])
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            dim=config["head_dim"], base=config["rope_theta"], mrope_section=params["mrope_section"]
        )

    def __call__(
        self,
        hidden: mx.array,
        positions: mx.array,
        image_indices: mx.array | None = None,
        deepstack: list[mx.array] | None = None,
    ) -> mx.array:
        rope = self.rotary_emb(hidden, positions)
        idx = mx.arange(hidden.shape[1])
        mask = (idx[:, None] >= idx[None, :])[None, None]
        for index, layer in enumerate(self.layers):
            hidden, _ = layer(hidden, attention_mask=mask, position_embeddings=rope)
            if deepstack is not None and index < len(deepstack):
                hidden[:, image_indices] += deepstack[index].astype(hidden.dtype)[None]
            mx.eval(hidden)
        # The diffusion checkpoint consumes the last decoder output before final RMSNorm.
        return hidden


class QwenImage21TextEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.language_model = LanguageModel(config["text_config"])
        vision = {
            key: value
            for key, value in config["vision_config"].items()
            if key in inspect.signature(Qwen3VLVisionModel).parameters
        }
        self.visual = Qwen3VLVisionModel(**vision, preserve_input_dtype=True)
        for block in self.visual.blocks:
            block.mlp.act_fn = self._tanh_gelu
        for merger in [self.visual.merger, *self.visual.deepstack_merger_list]:
            merger.act_fn = self._exact_gelu
        self.image_token_id = config["image_token_id"]
        self.merge_size = config["vision_config"]["spatial_merge_size"]

    def __call__(
        self, input_ids: mx.array, pixel_values: mx.array | None = None, image_grid_thw: mx.array | None = None
    ) -> mx.array:
        if input_ids.shape[0] != 1:
            raise ValueError("Qwen-Image-2.1 currently encodes one prompt at a time.")
        hidden = self.language_model.embed_tokens(input_ids)
        image_indices = mx.array(np.flatnonzero(np.asarray(input_ids[0]) == self.image_token_id), dtype=mx.int32)
        deepstack = None
        if pixel_values is not None:
            features, deepstack = self.visual(pixel_values.astype(hidden.dtype), image_grid_thw, return_deepstack=True)
            if features.shape[0] != image_indices.size:
                raise ValueError("Vision feature count does not match the image placeholders.")
            hidden[:, image_indices] = features.astype(hidden.dtype)[None]
        positions = self.position_ids(input_ids, image_grid_thw)
        return self.language_model(hidden, positions, image_indices, deepstack)

    def position_ids(self, input_ids: mx.array, grids: mx.array | None) -> mx.array:
        ids = np.asarray(input_ids[0])
        result = np.zeros((3, len(ids)), dtype=np.int32)
        cursor, position = 0, 0
        if grids is not None:
            for t, h, w in grids.tolist():
                h, w = h // self.merge_size, w // self.merge_size
                start = int(np.flatnonzero(ids[cursor:] == self.image_token_id)[0]) + cursor
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
    def _exact_gelu(value: mx.array) -> mx.array:
        return nn.gelu(value.astype(mx.float32)).astype(value.dtype)

    @staticmethod
    def _tanh_gelu(value: mx.array) -> mx.array:
        return nn.gelu_approx(value.astype(mx.float32)).astype(value.dtype)
