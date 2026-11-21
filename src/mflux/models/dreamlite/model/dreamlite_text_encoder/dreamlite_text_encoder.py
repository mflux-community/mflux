import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_language_model import Qwen3VLLanguageModel
from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel


class DreamLiteTextEncoder(nn.Module):
    # Qwen3-VL-2B with its vision tower: an edit shows the encoder the source image beside the instruction
    IMAGE_TOKEN_ID = 151655
    TEXT_CONFIG = {
        "vocab_size": 151936,
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 6144,
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
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": [5, 11, 17],
    }

    def __init__(self, text_config: dict | None = None, vision_config: dict | None = None):
        super().__init__()
        self.language_model = Qwen3VLLanguageModel(text_config or DreamLiteTextEncoder.TEXT_CONFIG)
        self.visual = Qwen3VLVisionModel(
            **(vision_config or DreamLiteTextEncoder.VISION_CONFIG), preserve_input_dtype=True
        )
        Qwen3VLMultimodal.use_reference_activations(self.visual)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> mx.array:
        model = self.language_model
        # The diffusion reads the last layer's states before the final norm, where bfloat16 rounding of 28 layers of
        # activations is visible in the image. The activations run in float32 over the stored weights instead.
        hidden = model.embed_tokens(input_ids).astype(mx.float32)
        image_indices = mx.array(np.flatnonzero(np.asarray(input_ids[0]) == DreamLiteTextEncoder.IMAGE_TOKEN_ID), dtype=mx.int32)  # fmt: off
        deepstack = None
        if pixel_values is not None:
            features, deepstack = self.visual(pixel_values.astype(hidden.dtype), image_grid_thw, return_deepstack=True)
            if features.shape[0] != image_indices.size:
                raise ValueError("Vision feature count does not match the image placeholders.")
            hidden[:, image_indices] = features.astype(hidden.dtype)[None]

        positions = Qwen3VLMultimodal.mrope_positions(
            input_ids, image_grid_thw, DreamLiteTextEncoder.IMAGE_TOKEN_ID, self.visual.spatial_merge_size
        )
        # transformers' hidden_states[-1]: the last decoder layer's output, before the final RMSNorm
        return model(hidden, positions, image_indices, deepstack)
