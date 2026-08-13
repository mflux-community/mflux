import pytest

from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig


class _DualTransformerModel:
    def __init__(self) -> None:
        self.conditional_transformer = object()
        self.unconditional_transformer = object()
        self.text_encoder = object()
        self.tiling_config = None


@pytest.mark.fast
def test_memory_saver_clears_dual_transformers_when_not_kept():
    model = _DualTransformerModel()
    saver = MemorySaver(model=model, keep_transformer=False)
    config = Config(
        width=256,
        height=256,
        guidance=7.0,
        scheduler="linear",
        model_config=ModelConfig.ideogram4_fp8(),
        num_inference_steps=1,
    )

    saver.call_after_loop(seed=1, prompt="{}", latents=None, config=config)

    assert model.conditional_transformer is None
    assert model.unconditional_transformer is None
