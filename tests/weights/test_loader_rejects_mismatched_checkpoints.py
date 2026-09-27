import mlx.core as mx
import pytest

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.utils.exceptions import ModelConfigError


class _TinyModel:
    MAPPING = [
        WeightTarget(to_pattern="proj.weight", from_pattern=["model.proj.weight"]),
        WeightTarget(to_pattern="extra.weight", from_pattern=["model.extra.weight"], required=False),
    ]

    @classmethod
    def get_components(cls) -> list[ComponentDefinition]:
        return [ComponentDefinition(name="encoder", hf_subdir="encoder", mapping_getter=lambda: cls.MAPPING)]

    @staticmethod
    def get_download_patterns() -> list[str]:
        return ["encoder/*.safetensors"]

    @staticmethod
    def write(root, tensors: dict[str, mx.array]) -> None:
        (root / "encoder").mkdir()
        mx.save_safetensors(str(root / "encoder" / "model.safetensors"), tensors)

    @classmethod
    def rejection(cls, root, tensors: dict[str, mx.array]) -> str:
        cls.write(root, tensors)
        with pytest.raises(ModelConfigError) as error:
            WeightLoader.load(weight_definition=cls, model_path=str(root))
        return str(error.value)


class _FiveLayerModel(_TinyModel):
    MAPPING = [WeightTarget(to_pattern=f"l{i}.weight", from_pattern=[f"model.l{i}.weight"]) for i in range(5)]


class _BlockModel(_TinyModel):
    MAPPING = [
        WeightTarget(to_pattern="kept.weight", from_pattern=["model.kept.weight"]),
        WeightTarget(to_pattern="blocks.{block}.proj.weight", from_pattern=["model.blocks.{block}.proj.weight"]),
    ]


class _QuantizedModel(_TinyModel):
    MAPPING = [
        WeightTarget(to_pattern="a.weight", from_pattern=["model.a.weight"]),
        WeightTarget(to_pattern="b.weight", from_pattern=["model.b.weight"]),
        WeightTarget(to_pattern="proj.weight", from_pattern=["model.proj.weight"]),
    ]


@pytest.mark.fast
def test_a_checkpoint_whose_names_match_nothing_is_rejected(tmp_path):
    message = _TinyModel.rejection(tmp_path, {"other.proj.weight": mx.ones((2, 2))})

    assert "encoder" in message
    assert "model.proj.weight" in message
    assert "other.proj.weight" in message
    assert str(tmp_path / "encoder") in message


@pytest.mark.fast
def test_the_message_counts_a_single_missing_weight_in_the_singular(tmp_path):
    message = _TinyModel.rejection(tmp_path, {"other.proj.weight": mx.ones((2, 2))})

    assert "1 required weight has no match" in message


@pytest.mark.fast
def test_a_checkpoint_with_every_required_weight_loads(tmp_path):
    _TinyModel.write(tmp_path, {"model.proj.weight": mx.ones((2, 2)), "unrelated.bias": mx.ones((2,))})

    weights = WeightLoader.load(weight_definition=_TinyModel, model_path=str(tmp_path))

    assert mx.array_equal(weights.components["encoder"]["proj"]["weight"], mx.ones((2, 2))).item()


@pytest.mark.fast
def test_the_message_counts_every_missing_weight_but_lists_three(tmp_path):
    message = _FiveLayerModel.rejection(tmp_path, {"other.weight": mx.ones((2,))})

    assert "5 required weights" in message
    assert sum(f"model.l{i}.weight" in message for i in range(5)) == 3


@pytest.mark.fast
def test_the_message_shows_the_renamed_tensor_not_the_ones_that_matched(tmp_path):
    # The issue 748 checkpoint matched 294 of 297 transformer weights; listing its first names in sort order would
    # show correctly named tensors and hide the renamed one the user has to see.
    message = _BlockModel.rejection(
        tmp_path, {"model.kept.weight": mx.ones((2,)), "model.blocks.0.renamed.weight": mx.ones((2,))}
    )

    assert "model.blocks.0.renamed.weight" in message
    assert "model.kept.weight" not in message
    assert "model.blocks.0.proj.weight" in message
    assert "{block}" not in message


@pytest.mark.fast
def test_the_message_names_the_block_that_lacks_a_weight(tmp_path):
    # Block 0 is complete; block 1 has another weight but not its projection, so its random start value would stay.
    message = _BlockModel.rejection(
        tmp_path,
        {
            "model.kept.weight": mx.ones((2,)),
            "model.blocks.0.proj.weight": mx.ones((2,)),
            "model.blocks.1.norm.weight": mx.ones((2,)),
        },
    )

    assert "1 required weight has no match (expected names like model.blocks.1.proj.weight;" in message


@pytest.mark.fast
def test_the_message_skips_quantization_scales_of_weights_that_matched(tmp_path):
    # The issue 748 checkpoint is 4-bit: 464 of its 467 unused names were the scales and biases of weights that
    # matched, and they sorted ahead of the three renamed tensors.
    quantized = {f"model.{layer}.{part}": mx.ones((2,)) for layer in "ab" for part in ("weight", "scales", "biases")}
    message = _QuantizedModel.rejection(tmp_path, quantized | {"model.renamed.weight": mx.ones((2,))})

    assert "model.renamed.weight" in message
    assert "scales" not in message
    assert "biases" not in message


@pytest.mark.fast
def test_a_quantized_checkpoint_missing_a_weight_outright_lists_its_weights_not_their_scales(tmp_path):
    quantized = {f"model.{layer}.{part}": mx.ones((2,)) for layer in "ab" for part in ("weight", "scales", "biases")}
    message = _QuantizedModel.rejection(tmp_path, quantized)

    assert "found model.a.weight, model.b.weight)" in message
