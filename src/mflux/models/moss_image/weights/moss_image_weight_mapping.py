from mflux.models.common.weights.mapping.weight_mapping import WeightTarget


class MOSSImageWeightMapping:
    @staticmethod
    def get_transformer_mapping() -> list[WeightTarget]:
        mapping = [
            WeightTarget("video_embedder.weight", ["video_embedder.weight"]),
            WeightTarget("video_embedder.bias", ["video_embedder.bias"]),
            WeightTarget("final_norm_video.weight", ["final_norm_video.weight"]),
            WeightTarget("final_linear_video.weight", ["final_linear_video.weight"]),
            WeightTarget("time_embedding_0.weight", ["time_embedding.0.weight"]),
            WeightTarget("time_embedding_0.bias", ["time_embedding.0.bias"]),
            WeightTarget("time_embedding_2.weight", ["time_embedding.2.weight"]),
            WeightTarget("time_embedding_2.bias", ["time_embedding.2.bias"]),
            WeightTarget("shared_block_modulation.weight", ["shared_block_modulation.weight"]),
            WeightTarget("shared_block_modulation.bias", ["shared_block_modulation.bias"]),
            WeightTarget("video_head_modulation.weight", ["video_head_modulation.weight"]),
            WeightTarget("video_head_modulation.bias", ["video_head_modulation.bias"]),
            WeightTarget("text_embedder.proj.weight", ["text_embedder.proj.weight"]),
            WeightTarget("text_embedder.proj.bias", ["text_embedder.proj.bias"]),
        ]

        # TextBlock (2 blocks)
        for b in range(2):
            for key in ["norm1.weight", "norm2.weight", "q_norm.weight", "k_norm.weight"]:
                mapping.append(WeightTarget(f"text_embedder.blocks.{b}.{key}", [f"text_embedder.blocks.{b}.{key}"]))
            for key in ["qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"]:
                mapping.append(WeightTarget(f"text_embedder.blocks.{b}.{key}.weight", [f"text_embedder.blocks.{b}.{key}.weight"]))
                mapping.append(WeightTarget(f"text_embedder.blocks.{b}.{key}.bias", [f"text_embedder.blocks.{b}.{key}.bias"]))

        # Layers (42 blocks)
        for i in range(42):
            for k in ["norm1.weight", "norm2.weight", "q_norm.weight", "k_norm.weight"]:
                mapping.append(WeightTarget(f"layers.{i}.{k}", [f"layers.{i}.{k}"]))
            for sub in ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]:
                mapping.append(WeightTarget(f"layers.{i}.proj_{sub}.weight", [f"layers.{i}.proj.{sub}.weight"]))

        return mapping
