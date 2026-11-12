from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms


class SenseNovaWeightMapping(WeightMapping):
    # The upstream checkpoint holds two streams per layer: the und tensors and their *_mot_gen twins. The gen-stream
    # transformer drops the suffix, so the text encoder can fill the same module names from the und tensors.
    _LM = "language_model.model"
    _FM = "fm_modules"

    _LAYER = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.q_norm",
        "self_attn.q_norm_hw",
        "self_attn.k_norm",
        "self_attn.k_norm_hw",
        "input_layernorm",
        "post_attention_layernorm",
    ]  # fmt: skip

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        lm = SenseNovaWeightMapping._LM
        pairs = [("embed_tokens.weight", f"{lm}.embed_tokens.weight")]
        pairs += [(f"layers.{{layer}}.{name}.weight", f"{lm}.layers.{{layer}}.{name}.weight") for name in SenseNovaWeightMapping._LAYER]  # fmt: skip
        pairs += [(f"layers.{{layer}}.mlp.{name}.weight", f"{lm}.layers.{{layer}}.mlp.{name}.weight") for name in ("gate_proj", "up_proj", "down_proj")]  # fmt: skip
        return [WeightTarget(to_pattern=to_name, from_pattern=[from_name]) for to_name, from_name in pairs]

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        lm, fm = SenseNovaWeightMapping._LM, SenseNovaWeightMapping._FM
        embeddings = f"{fm}.vision_model_mot_gen.embeddings"
        pairs = [("norm.weight", f"{lm}.norm_mot_gen.weight")]
        convs = []
        for name in ("conv1", "conv2"):
            pairs.append((f"head.{name}.bias", f"{fm}.fm_head.{name}.bias"))
            convs.append((f"head.{name}.weight", f"{fm}.fm_head.{name}.weight"))
        for name in ("patch_embedding", "dense_embedding"):
            pairs.append((f"patch_embedder.{name}.bias", f"{embeddings}.{name}.bias"))
            convs.append((f"patch_embedder.{name}.weight", f"{embeddings}.{name}.weight"))
        for name in ("timestep_embedder", "noise_scale_embedder"):
            for index in (0, 2):
                pairs += [(f"{name}.mlp.{index}.{part}", f"{fm}.{name}.mlp.{index}.{part}") for part in ("weight", "bias")]  # fmt: skip
        pairs += [(f"layers.{{layer}}.{name}.weight", f"{lm}.layers.{{layer}}.{name}_mot_gen.weight") for name in SenseNovaWeightMapping._LAYER]  # fmt: skip
        pairs += [(f"layers.{{layer}}.mlp.{name}.weight", f"{lm}.layers.{{layer}}.mlp_mot_gen.{name}.weight") for name in ("gate_proj", "up_proj", "down_proj")]  # fmt: skip

        targets = [WeightTarget(to_pattern=to_name, from_pattern=[from_name]) for to_name, from_name in pairs]
        transform = WeightTransforms.transpose_conv2d_weight
        return targets + [WeightTarget(to_pattern=to_name, from_pattern=[from_name], transform=transform) for to_name, from_name in convs]  # fmt: skip
