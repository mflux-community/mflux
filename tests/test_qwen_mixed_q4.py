from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition


class SmallBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_mod_linear = nn.Linear(64, 384)
        self.attn_proj = nn.Linear(64, 64)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = [SmallBlock(), SmallBlock()]
        self.proj_out = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(dims=64, affine=False)  # not quantizable


@pytest.mark.fast
class TestQwenMixedQuantizationPredicate:
    def test_img_mod_linear_is_8bit_at_q4(self):
        module = nn.Linear(64, 384)
        result = QwenWeightDefinition.quantization_predicate("transformer_blocks.0.img_mod_linear", module, bits=4)
        assert result == {"bits": 8}

    def test_img_mod_linear_follows_global_level_at_q8(self):
        module = nn.Linear(64, 384)
        result = QwenWeightDefinition.quantization_predicate("transformer_blocks.0.img_mod_linear", module, bits=8)
        assert result is True

    def test_other_layers_follow_global_level_at_q4(self):
        module = nn.Linear(64, 64)
        assert QwenWeightDefinition.quantization_predicate("transformer_blocks.0.attn_proj", module, bits=4) is True

    def test_non_quantizable_module_is_skipped(self):
        module = nn.LayerNorm(dims=64, affine=False)
        assert QwenWeightDefinition.quantization_predicate("norm", module, bits=4) is False


@pytest.mark.fast
class TestPredicateWithBits:
    def test_two_arg_predicate_passes_through_unchanged(self):
        def predicate(path, module):
            return True

        assert WeightApplier._predicate_with_bits(predicate, 4) is predicate

    def test_three_arg_predicate_receives_bits(self):
        seen = {}

        def predicate(path, module, bits=None):
            seen["bits"] = bits
            return True

        wrapped = WeightApplier._predicate_with_bits(predicate, 4)
        assert wrapped is not predicate
        wrapped("p", object())
        assert seen["bits"] == 4

    def test_none_predicate_stays_none(self):
        assert WeightApplier._predicate_with_bits(None, 4) is None


@pytest.mark.fast
class TestStoredLayerPredicate:
    @staticmethod
    def _mixed_quantize(model):
        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if ".img_mod_linear" in path:
                return {"bits": 8}
            return True

        nn.quantize(model, bits=4, class_predicate=predicate)

    def test_mixed_save_reconstructs_per_layer_bits(self):
        saved = SmallModel()
        self._mixed_quantize(saved)
        component_weights = saved.parameters()

        fresh = SmallModel()
        stored_predicate = WeightApplier._stored_layer_predicate(component_weights, None)
        nn.quantize(fresh, bits=4, class_predicate=stored_predicate)

        for block in fresh.transformer_blocks:
            assert block.img_mod_linear.bits == 8
            assert block.attn_proj.bits == 4
        assert fresh.proj_out.bits == 4

        # The reconstructed structure must accept the stored weights and
        # produce identical outputs.
        fresh.update(component_weights, strict=False)
        x = mx.random.normal((2, 64))
        expected = saved.transformer_blocks[0].img_mod_linear(x)
        actual = fresh.transformer_blocks[0].img_mod_linear(x)
        assert mx.array_equal(expected, actual)

    def test_uniform_save_keeps_uniform_bits(self):
        saved = SmallModel()
        nn.quantize(saved, bits=4, class_predicate=lambda p, m: hasattr(m, "to_quantized"))
        component_weights = saved.parameters()

        fresh = SmallModel()
        stored_predicate = WeightApplier._stored_layer_predicate(component_weights, None)
        nn.quantize(fresh, bits=4, class_predicate=stored_predicate)

        for block in fresh.transformer_blocks:
            assert block.img_mod_linear.bits == 4
            assert block.attn_proj.bits == 4

    def test_unquantized_layer_in_save_stays_unquantized(self):
        saved = SmallModel()

        def skip_proj(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            return path != "proj_out"

        nn.quantize(saved, bits=4, class_predicate=skip_proj)
        component_weights = saved.parameters()

        fresh = SmallModel()
        stored_predicate = WeightApplier._stored_layer_predicate(component_weights, None)
        nn.quantize(fresh, bits=4, class_predicate=stored_predicate)

        assert not hasattr(fresh.proj_out, "bits")
        fresh.update(component_weights, strict=False)
        x = mx.random.normal((2, 64))
        assert mx.array_equal(saved.proj_out(x), fresh.proj_out(x))

    def test_base_predicate_still_gates_quantizability(self):
        saved = SmallModel()
        self._mixed_quantize(saved)
        component_weights = saved.parameters()

        fresh = SmallModel()
        stored_predicate = WeightApplier._stored_layer_predicate(
            component_weights, lambda p, m: hasattr(m, "to_quantized")
        )
        nn.quantize(fresh, bits=4, class_predicate=stored_predicate)
        assert not hasattr(fresh.norm, "bits")


@pytest.mark.fast
class TestStoredLayerGroupSizes:
    def test_group_size_32_save_reconstructs_exactly(self):
        saved = SmallModel()
        nn.quantize(saved, bits=4, group_size=32, class_predicate=lambda p, m: hasattr(m, "to_quantized"))
        component_weights = saved.parameters()

        fresh = SmallModel()
        stored_predicate = WeightApplier._stored_layer_predicate(component_weights, None)
        nn.quantize(fresh, bits=4, class_predicate=stored_predicate)

        for block in fresh.transformer_blocks:
            assert block.img_mod_linear.bits == 4
            assert block.img_mod_linear.group_size == 32
        fresh.update(component_weights, strict=False)
        x = mx.random.normal((2, 64))
        assert mx.array_equal(saved.proj_out(x), fresh.proj_out(x))

    def test_mixed_group_sizes_reconstruct_per_layer(self):
        saved = SmallModel()

        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if ".img_mod_linear" in path:
                return {"bits": 8, "group_size": 32}
            return True

        nn.quantize(saved, bits=4, class_predicate=predicate)
        component_weights = saved.parameters()

        fresh = SmallModel()
        stored_predicate = WeightApplier._stored_layer_predicate(component_weights, None)
        nn.quantize(fresh, bits=4, class_predicate=stored_predicate)

        block = fresh.transformer_blocks[0]
        assert block.img_mod_linear.bits == 8
        assert block.img_mod_linear.group_size == 32
        assert block.attn_proj.bits == 4
        assert block.attn_proj.group_size == 64
        fresh.update(component_weights, strict=False)
        x = mx.random.normal((2, 64))
        expected = saved.transformer_blocks[0].img_mod_linear(x)
        actual = fresh.transformer_blocks[0].img_mod_linear(x)
        assert mx.array_equal(expected, actual)

    def test_unparseable_shapes_fall_back_to_global_level(self):
        saved = SmallModel()
        nn.quantize(saved, bits=4, class_predicate=lambda p, m: hasattr(m, "to_quantized"))
        component_weights = saved.parameters()
        # Corrupt one layer's scales so neither bits nor group size resolve.
        component_weights["proj_out"]["scales"] = mx.zeros((64, 3))

        fresh = SmallModel()
        stored_predicate = WeightApplier._stored_layer_predicate(component_weights, None)
        nn.quantize(fresh, bits=4, class_predicate=stored_predicate)
        assert fresh.proj_out.bits == 4


@pytest.mark.fast
class TestQuantizeWiring:
    """The helpers are covered above; this is the path a load actually takes."""

    @staticmethod
    def _definition(predicate=None):
        return SimpleNamespace(quantization_predicate=predicate or (lambda p, m: hasattr(m, "to_quantized")))

    def test_quantize_rebuilds_a_mixed_save_at_its_stored_precision(self):
        saved = SmallModel()
        saved_predicate = WeightApplier._predicate_with_bits(QwenWeightDefinition.quantization_predicate, 4)
        nn.quantize(saved, bits=4, class_predicate=saved_predicate)
        weights = LoadedWeights(components={"transformer": saved.parameters()}, meta_data=MetaData())

        fresh = SmallModel()
        WeightApplier._quantize(
            models={"transformer": fresh},
            bits=4,
            components={},
            weight_definition=self._definition(QwenWeightDefinition.quantization_predicate),
            weights=weights,
        )

        block = fresh.transformer_blocks[0]
        assert (block.img_mod_linear.bits, block.attn_proj.bits) == (8, 4)

    def test_quantize_honours_skip_quantization(self):
        fresh = SmallModel()
        component = SimpleNamespace(skip_quantization=True, weight_subkey=None)
        WeightApplier._quantize(
            models={"transformer": fresh},
            bits=4,
            components={"transformer": component},
            weight_definition=self._definition(),
            weights=None,
        )

        assert not isinstance(fresh.proj_out, nn.QuantizedLinear)


@pytest.mark.fast
def test_a_keyword_only_third_parameter_is_not_treated_as_bits():
    # A predicate whose third parameter cannot take a positional argument must be left
    # alone: calling it as predicate(path, module, bits) would raise TypeError.
    def predicate(path, module, *, debug=False):
        return True

    assert WeightApplier._predicate_with_bits(predicate, 4) is predicate
