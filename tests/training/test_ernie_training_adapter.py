from types import SimpleNamespace

import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.ernie_image.training_adapter import ernie_training_adapter as ernie_adapter_module


class _FakeErnieImage:
    def __init__(self, *, quantize, model_config, model_path):
        self.transformer = SimpleNamespace()
        self.last_generate_kwargs = None

    def generate_image(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return SimpleNamespace(image="preview-image")


@pytest.mark.fast
def test_ernie_turbo_preview_uses_canonical_guidance_and_steps(monkeypatch):
    monkeypatch.setattr(ernie_adapter_module, "ErnieImage", _FakeErnieImage)

    adapter = ernie_adapter_module.ErnieTrainingAdapter(model_config=ModelConfig.ernie_image_turbo(), quantize=None)
    training_spec = SimpleNamespace(steps=4, guidance=9.0)
    adapter.create_config(training_spec=training_spec, width=512, height=512)
    adapter.generate_preview_image(seed=0, prompt="test", width=512, height=512, steps=4)

    assert adapter._ernie.last_generate_kwargs["guidance"] == 1.0
    assert adapter._ernie.last_generate_kwargs["num_inference_steps"] == 8


@pytest.mark.fast
def test_ernie_base_preview_uses_canonical_guidance_and_steps(monkeypatch):
    monkeypatch.setattr(ernie_adapter_module, "ErnieImage", _FakeErnieImage)

    adapter = ernie_adapter_module.ErnieTrainingAdapter(model_config=ModelConfig.ernie_image(), quantize=None)
    training_spec = SimpleNamespace(steps=10, guidance=9.0)
    adapter.create_config(training_spec=training_spec, width=512, height=512)
    adapter.generate_preview_image(seed=0, prompt="test", width=512, height=512, steps=10)

    assert adapter._ernie.last_generate_kwargs["guidance"] == 4.0
    assert adapter._ernie.last_generate_kwargs["num_inference_steps"] == 50
