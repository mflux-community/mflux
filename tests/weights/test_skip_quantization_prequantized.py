from types import SimpleNamespace

import pytest
from mlx import nn

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier

_DEFINITION = SimpleNamespace(quantization_predicate=None)
_SKIP = {"text_encoder": SimpleNamespace(skip_quantization=True, weight_subkey=None)}


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(64, 64))


@pytest.mark.fast
class TestSkipQuantizationWithPrequantizedCheckpoint:
    def test_on_load_quantization_keeps_skip(self):
        model = _model()
        WeightApplier._quantize({"text_encoder": model}, 8, _SKIP, _DEFINITION, weights=None)
        assert isinstance(model.layers[0], nn.Linear)

    def test_component_missing_from_checkpoint_keeps_skip(self):
        model = _model()
        weights = LoadedWeights(components={"transformer": {}}, meta_data=None)
        WeightApplier._quantize({"text_encoder": model}, 8, _SKIP, _DEFINITION, weights=weights)
        assert isinstance(model.layers[0], nn.Linear)

    def test_quantized_component_in_checkpoint_is_structured_as_quantized(self):
        saved = _model()
        nn.quantize(saved, bits=8)
        stored = dict(saved.parameters())
        model = _model()
        weights = LoadedWeights(components={"text_encoder": stored}, meta_data=None)
        WeightApplier._quantize({"text_encoder": model}, 8, _SKIP, _DEFINITION, weights=weights)
        assert isinstance(model.layers[0], nn.QuantizedLinear)

    def test_unquantized_component_in_checkpoint_stays_unquantized(self):
        stored = dict(_model().parameters())
        model = _model()
        weights = LoadedWeights(components={"text_encoder": stored}, meta_data=None)
        WeightApplier._quantize({"text_encoder": model}, 8, _SKIP, _DEFINITION, weights=weights)
        assert isinstance(model.layers[0], nn.Linear)
