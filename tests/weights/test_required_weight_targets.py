import mlx.core as mx
import pytest

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget


class _Weights:
    @staticmethod
    def named(*names: str) -> dict[str, mx.array]:
        return {name: mx.zeros((1,)) for name in names}


@pytest.mark.fast
def test_a_required_target_with_no_source_tensor_is_reported():
    mapping = [
        WeightTarget(to_pattern="a.weight", from_pattern=["a.weight"]),
        WeightTarget(to_pattern="b.weight", from_pattern=["b.weight"]),
    ]

    missing = WeightMapper.missing_required_targets(_Weights.named("a.weight"), mapping)

    assert [target.to_pattern for target in missing] == ["b.weight"]


@pytest.mark.fast
def test_an_optional_target_is_never_reported():
    mapping = [WeightTarget(to_pattern="b.weight", from_pattern=["b.weight"], required=False)]

    assert WeightMapper.missing_required_targets(_Weights.named("a.weight"), mapping) == []


@pytest.mark.fast
def test_one_matching_block_satisfies_a_block_pattern():
    # The mapper expands every block pattern to the largest block count it detects, so FLUX.1's 19 double
    # blocks are looked for up to 38. Counting phantom blocks as missing would reject every FLUX.1 checkpoint.
    mapping = [
        WeightTarget(
            to_pattern="transformer_blocks.{block}.attn.to_q.weight",
            from_pattern=["transformer_blocks.{block}.attn.to_q.weight"],
        )
    ]
    weights = _Weights.named("transformer_blocks.0.attn.to_q.weight", "single_transformer_blocks.37.proj_out.weight")

    assert WeightMapper.missing_required_targets(weights, mapping) == []


@pytest.mark.fast
def test_any_alternative_source_name_satisfies_a_target():
    mapping = [WeightTarget(to_pattern="norm.weight", from_pattern=["old.norm.weight", "new.norm.weight"])]

    assert WeightMapper.missing_required_targets(_Weights.named("new.norm.weight"), mapping) == []


class _Blocks:
    MAPPING = [
        WeightTarget(to_pattern="blocks.{block}.a.weight", from_pattern=["blocks.{block}.a.weight"]),
        WeightTarget(to_pattern="blocks.{block}.b.weight", from_pattern=["blocks.{block}.b.weight"]),
        WeightTarget(to_pattern="blocks.{block}.bias", from_pattern=["blocks.{block}.bias"], required=False),
    ]


@pytest.mark.fast
def test_a_block_that_lacks_a_required_weight_is_reported_even_when_another_block_has_it():
    weights = _Weights.named("blocks.0.a.weight", "blocks.0.b.weight", "blocks.1.a.weight")

    assert WeightMapper.missing_required_names(weights, _Blocks.MAPPING) == ["blocks.1.b.weight"]


@pytest.mark.fast
def test_an_optional_block_weight_that_only_some_blocks_carry_is_not_reported():
    # The FLUX.1 VAE's last up block has no upsampler; its optional entry must not demand one in every block.
    weights = _Weights.named("blocks.0.a.weight", "blocks.0.b.weight", "blocks.0.bias")
    weights |= _Weights.named("blocks.1.a.weight", "blocks.1.b.weight")

    assert WeightMapper.missing_required_names(weights, _Blocks.MAPPING) == []


@pytest.mark.fast
def test_a_layer_that_lacks_a_required_weight_is_reported():
    mapping = [
        WeightTarget(
            to_pattern="layers.{layer}.q_proj.weight", from_pattern=["model.layers.{layer}.self_attn.q_proj.weight"]
        )
    ]
    weights = _Weights.named("model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.up_proj.weight")

    assert WeightMapper.missing_required_names(weights, mapping) == ["model.layers.1.self_attn.q_proj.weight"]


@pytest.mark.fast
def test_a_block_weight_capped_by_max_blocks_is_not_demanded_from_later_blocks():
    # Only the first Z-Image control layer has before_proj; max_blocks=1 says so.
    mapping = [
        WeightTarget(
            to_pattern="layers.{block}.before_proj.weight",
            from_pattern=["layers.{block}.before_proj.weight"],
            max_blocks=1,
        )
    ]
    weights = _Weights.named("layers.0.before_proj.weight", "layers.0.proj.weight", "layers.1.proj.weight")

    assert WeightMapper.missing_required_names(weights, mapping) == []


@pytest.mark.fast
def test_a_mid_block_resnet_that_lacks_a_required_weight_is_reported():
    mapping = [
        WeightTarget(
            to_pattern="mid.resnets.{i}.conv1.weight", from_pattern=["decoder.mid_block.resnets.{i}.conv1.weight"]
        )
    ]
    weights = _Weights.named("decoder.mid_block.resnets.0.conv1.weight", "decoder.mid_block.resnets.1.conv2.weight")

    assert WeightMapper.missing_required_names(weights, mapping) == ["decoder.mid_block.resnets.1.conv1.weight"]


@pytest.mark.fast
def test_an_up_block_resnet_that_lacks_a_required_weight_is_reported():
    mapping = [
        WeightTarget(
            to_pattern="up.{block}.resnets.{res}.conv1.weight",
            from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv1.weight"],
        )
    ]
    weights = _Weights.named(
        "decoder.up_blocks.0.resnets.0.conv1.weight",
        "decoder.up_blocks.1.resnets.0.conv1.weight",
        "decoder.up_blocks.1.resnets.1.conv2.weight",
    )

    assert WeightMapper.missing_required_names(weights, mapping) == ["decoder.up_blocks.1.resnets.1.conv1.weight"]


@pytest.mark.fast
def test_an_attention_index_the_checkpoint_does_not_have_is_not_demanded():
    # The mapper expands {i} to two, but a VAE mid block has a single attention.
    mapping = [
        WeightTarget(
            to_pattern="mid.attentions.{i}.to_q.weight", from_pattern=["decoder.mid_block.attentions.{i}.to_q.weight"]
        )
    ]
    weights = _Weights.named("decoder.mid_block.attentions.0.to_q.weight")

    assert WeightMapper.missing_required_names(weights, mapping) == []


class _OptionalFamily:
    MAPPING = [
        WeightTarget(
            to_pattern="single.{block}.proj.weight",
            from_pattern=["single.{block}.proj.weight"],
            required=False,
            complete_when_present=True,
        )
    ]


@pytest.mark.fast
def test_an_optional_block_family_the_checkpoint_does_not_have_is_not_reported():
    assert WeightMapper.missing_required_names(_Weights.named("double.0.proj.weight"), _OptionalFamily.MAPPING) == []


@pytest.mark.fast
def test_an_optional_block_family_the_checkpoint_has_must_be_complete():
    weights = _Weights.named("single.0.proj.weight", "single.1.norm.weight")

    assert WeightMapper.missing_required_names(weights, _OptionalFamily.MAPPING) == ["single.1.proj.weight"]
