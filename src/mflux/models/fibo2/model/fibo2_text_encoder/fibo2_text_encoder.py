import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_language_model import Qwen3VLLanguageModel
from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel


class Fibo2TextEncoder(nn.Module):
    # Qwen3-VL-4B with its vision tower; FIBO 2 reads hidden states up to index 35, so the last layer is not built
    NUM_LAYERS = 35
    IMAGE_TOKEN_ID = 151655
    TEXT_CONFIG = {
        "vocab_size": 151936,
        "hidden_size": 2560,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "rope_theta": 5000000.0,
        "rope_scaling": {"mrope_section": [24, 20, 20]},
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
    }
    VISION_CONFIG = {
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": [5, 11, 17],
    }

    def __init__(self, text_config: dict | None = None, vision_config: dict | None = None):
        super().__init__()
        self.language_model = Qwen3VLLanguageModel(text_config or Fibo2TextEncoder.TEXT_CONFIG)
        self.visual = Qwen3VLVisionModel(**(vision_config or Fibo2TextEncoder.VISION_CONFIG), preserve_input_dtype=True)
        Qwen3VLMultimodal.use_reference_activations(self.visual)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> list[mx.array]:
        model = self.language_model
        hidden = model.embed_tokens(input_ids)
        image_indices = mx.array(np.flatnonzero(np.asarray(input_ids[0]) == Fibo2TextEncoder.IMAGE_TOKEN_ID), dtype=mx.int32)  # fmt: off
        deepstack = None
        if pixel_values is not None:
            features, deepstack = self.visual(pixel_values.astype(hidden.dtype), image_grid_thw, return_deepstack=True)
            if features.shape[0] != image_indices.size:
                raise ValueError("Vision feature count does not match the image placeholders.")
            hidden[:, image_indices] = features.astype(hidden.dtype)[None]

        positions = Qwen3VLMultimodal.mrope_positions(
            input_ids, image_grid_thw, Fibo2TextEncoder.IMAGE_TOKEN_ID, self.visual.spatial_merge_size
        )
        return model.hidden_states(hidden, positions, image_indices, deepstack)
