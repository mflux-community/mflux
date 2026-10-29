from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.moss_image.moss_image_initializer import MOSSImageInitializer
from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition

pytestmark = pytest.mark.fast


class TinyMOSS:
    @staticmethod
    def make() -> SimpleNamespace:
        # The text encoder is the largest part of the real checkpoint; here it is the smaller one,
        # so a read of it alone is told apart from a read of everything.
        return SimpleNamespace(
            vae=nn.Sequential(nn.Linear(256, 256, bias=False)),
            transformer=nn.Sequential(nn.Linear(1024, 4096, bias=False)),
            text_encoder=nn.Sequential(nn.Linear(1024, 1024, bias=False)),
        )

    @staticmethod
    def init_models(model, model_config) -> None:
        for name, module in vars(TinyMOSS.make()).items():
            setattr(model, name, module)


def _nbytes(module: nn.Module) -> int:
    return sum(value.nbytes for _, value in tree_flatten(module.parameters()))


@pytest.mark.parametrize(("saved_bits", "quantize"), [(None, None), (None, 8), (8, None)])
def test_load_reads_each_component_at_its_first_use(tmp_path, monkeypatch, saved_bits, quantize):
    # Evaluating every parameter at load held the text encoder, transformer and VAE at once, so
    # --low-ram could only free the text encoder after that peak (#948). Measured as memory, so a
    # read through any path counts: nothing is read at load, and encoding reads the text encoder alone.
    original = TinyMOSS.make()
    if saved_bits:
        for module in vars(original).values():
            nn.quantize(module, bits=saved_bits, class_predicate=MOSSImageWeightDefinition.quantization_predicate)
    x = mx.ones((1, 1024))
    expected = original.text_encoder(x)
    mx.eval([module.parameters() for module in vars(original).values()], expected)
    ModelSaver.save_model(original, saved_bits, str(tmp_path), MOSSImageWeightDefinition)
    monkeypatch.setattr(MOSSImageInitializer, "_init_models", TinyMOSS.init_models)
    monkeypatch.setattr(MOSSImageInitializer, "_init_tokenizers", lambda model, path: None)
    restored = nn.Module()
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()

    MOSSImageInitializer.init(restored, ModelConfig.moss_image_sft(), quantize=quantize, model_path=str(tmp_path))

    assert mx.get_active_memory() - before < 1e6
    encoded = restored.text_encoder(x)
    mx.eval(encoded)
    # The dense weights that -q quantizes are released when the GPU finishes, a moment after eval returns.
    mx.synchronize()
    read = mx.get_active_memory() - before
    text_encoder_bytes = _nbytes(restored.text_encoder)
    assert text_encoder_bytes <= read < text_encoder_bytes + _nbytes(restored.transformer)
    assert mx.allclose(encoded, expected, atol=0.05).item()
