import pytest
from mlx.utils import tree_flatten

from mflux.models.sensenova.model.sensenova_text_encoder.sensenova_text_encoder import SenseNovaTextEncoder
from mflux.models.sensenova.model.sensenova_transformer.sensenova_transformer import SenseNovaTransformer
from mflux.models.sensenova.weights.sensenova_weight_definition import SenseNovaWeightDefinition
from mflux.models.sensenova.weights.sensenova_weight_mapping import SenseNovaWeightMapping

NUM_LAYERS = 42
TRANSFORMER_TENSORS = 563
TEXT_ENCODER_TENSORS = 547
LM = "language_model.model"


@pytest.mark.fast
class TestSenseNovaWeightMapping:
    def test_every_transformer_parameter_is_loaded_exactly_once_and_nothing_else(self):
        targets = SenseNovaWeightMapping.get_transformer_mapping()
        parameters = {name for name, _ in tree_flatten(SenseNovaTransformer().parameters())}
        expanded = {target.to_pattern.format(layer=i) for target in targets for i in range(NUM_LAYERS)} | {
            target.to_pattern for target in targets if "{layer}" not in target.to_pattern
        }

        assert expanded == parameters
        assert len(parameters) == TRANSFORMER_TENSORS

    def test_every_text_encoder_parameter_is_loaded_exactly_once_and_nothing_else(self):
        targets = SenseNovaWeightMapping.get_text_encoder_mapping()
        parameters = {name for name, _ in tree_flatten(SenseNovaTextEncoder().parameters())}
        expanded = {target.to_pattern.format(layer=i) for target in targets for i in range(NUM_LAYERS)} | {
            target.to_pattern for target in targets if "{layer}" not in target.to_pattern
        }

        assert expanded == parameters
        assert len(parameters) == TEXT_ENCODER_TENSORS

    def test_the_gen_stream_reads_the_mot_gen_tensors_and_the_und_stream_the_plain_ones(self):
        gen = {t.to_pattern: t.from_pattern[0] for t in SenseNovaWeightMapping.get_transformer_mapping()}
        und = {t.to_pattern: t.from_pattern[0] for t in SenseNovaWeightMapping.get_text_encoder_mapping()}

        assert gen["layers.{layer}.self_attn.q_proj.weight"] == f"{LM}.layers.{{layer}}.self_attn.q_proj_mot_gen.weight"
        assert gen["layers.{layer}.self_attn.q_norm_hw.weight"] == f"{LM}.layers.{{layer}}.self_attn.q_norm_hw_mot_gen.weight"  # fmt: skip
        assert gen["layers.{layer}.mlp.up_proj.weight"] == f"{LM}.layers.{{layer}}.mlp_mot_gen.up_proj.weight"
        assert gen["norm.weight"] == f"{LM}.norm_mot_gen.weight"
        assert und["layers.{layer}.self_attn.q_proj.weight"] == f"{LM}.layers.{{layer}}.self_attn.q_proj.weight"
        assert und["layers.{layer}.mlp.up_proj.weight"] == f"{LM}.layers.{{layer}}.mlp.up_proj.weight"
        assert und["embed_tokens.weight"] == f"{LM}.embed_tokens.weight"

    def test_nn_sequential_indices_are_renamed_and_conv_weights_transposed(self):
        targets = {t.to_pattern: t for t in SenseNovaWeightMapping.get_transformer_mapping()}

        assert targets["timestep_embedder.mlp.2.bias"].from_pattern == ["fm_modules.timestep_embedder.mlp.2.bias"]
        assert targets["noise_scale_embedder.mlp.0.weight"].transform is None
        for name in ("head.conv1.weight", "head.conv2.weight", "patch_embedder.patch_embedding.weight", "patch_embedder.dense_embedding.weight"):  # fmt: skip
            assert targets[name].transform is not None, name
        assert targets["patch_embedder.dense_embedding.weight"].from_pattern == [
            "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight"
        ]


@pytest.mark.fast
class TestSenseNovaWeightDefinition:
    def test_the_transformer_keeps_the_gen_stream_and_the_fm_modules(self):
        keep = SenseNovaWeightDefinition.source_transformer().key_transform

        for key in (
            f"{LM}.layers.0.self_attn.q_proj_mot_gen.weight",
            f"{LM}.layers.41.mlp_mot_gen.down_proj.weight",
            f"{LM}.layers.3.input_layernorm_mot_gen.weight",
            f"{LM}.norm_mot_gen.weight",
            "fm_modules.fm_head.conv1.weight",
            "fm_modules.timestep_embedder.mlp.0.weight",
            "fm_modules.noise_scale_embedder.mlp.2.bias",
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        ):
            assert keep(key), key

    def test_the_transformer_drops_the_und_stream_the_head_and_the_vision_tower(self):
        keep = SenseNovaWeightDefinition.source_transformer().key_transform

        for key in (
            f"{LM}.layers.0.self_attn.q_proj.weight",
            f"{LM}.layers.0.mlp.gate_proj.weight",
            f"{LM}.embed_tokens.weight",
            f"{LM}.norm.weight",
            "language_model.lm_head.weight",
            "vision_model.embeddings.patch_embedding.weight",
        ):
            assert not keep(key), key

    def test_the_text_encoder_keeps_the_und_stream_and_the_embeddings_only(self):
        keep = SenseNovaWeightDefinition.source_text_encoder().key_transform

        for key in (f"{LM}.embed_tokens.weight", f"{LM}.layers.0.self_attn.k_norm_hw.weight", f"{LM}.layers.41.mlp.up_proj.weight"):  # fmt: skip
            assert keep(key), key
        for key in (
            f"{LM}.layers.0.self_attn.k_norm_hw_mot_gen.weight",
            f"{LM}.layers.0.mlp_mot_gen.up_proj.weight",
            f"{LM}.norm.weight",
            f"{LM}.norm_mot_gen.weight",
            "language_model.lm_head.weight",
            "vision_model.embeddings.patch_embedding.weight",
            "fm_modules.fm_head.conv1.weight",
        ):
            assert not keep(key), key

    def test_a_truncated_model_drops_the_layers_past_its_depth(self):
        gen = SenseNovaWeightDefinition.source_transformer(4).key_transform
        und = SenseNovaWeightDefinition.source_text_encoder(4).key_transform

        assert gen(f"{LM}.layers.3.self_attn.q_proj_mot_gen.weight")
        assert not gen(f"{LM}.layers.4.self_attn.q_proj_mot_gen.weight")
        assert und(f"{LM}.layers.3.self_attn.q_proj.weight")
        assert not und(f"{LM}.layers.4.self_attn.q_proj.weight")

    def test_the_saved_layout_is_transformer_text_encoder_and_tokenizer(self):
        assert [c.name for c in SenseNovaWeightDefinition.get_components()] == ["transformer", "text_encoder"]
        assert [t.hf_subdir for t in SenseNovaWeightDefinition.get_tokenizers()] == ["tokenizer"]


@pytest.mark.fast
class TestSenseNovaQuantizationPredicate:
    @staticmethod
    def _quantized_paths() -> set[str]:
        model = SenseNovaTransformer(num_layers=1)
        return {
            path
            for path, module in model.named_modules()
            if SenseNovaWeightDefinition.quantization_predicate(path, module)
        }

    def test_the_linears_of_a_layer_and_the_patch_free_parts_quantize(self):
        paths = TestSenseNovaQuantizationPredicate._quantized_paths()

        for expected in ("layers.0.self_attn.q_proj", "layers.0.self_attn.o_proj", "layers.0.mlp.down_proj"):
            assert expected in paths, expected

    def test_the_conditioning_embedders_and_the_convolutions_stay_dense(self):
        paths = TestSenseNovaQuantizationPredicate._quantized_paths()

        assert not any(path.startswith(("timestep_embedder", "noise_scale_embedder")) for path in paths)
        assert not any(path.startswith(("head.", "patch_embedder.")) for path in paths)
