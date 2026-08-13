from unittest.mock import patch

import pytest

from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig


class _EncoderModel:
    def __init__(self, *, prompt_cache: dict | None = None) -> None:
        self.text_encoder = object()
        self.transformer = object()
        self.tiling_config = None
        self.prompt_cache = {} if prompt_cache is None else prompt_cache


def _config() -> Config:
    return Config(
        width=256,
        height=256,
        guidance=1.0,
        scheduler="linear",
        model_config=ModelConfig.flux2_klein_4b(),
        num_inference_steps=1,
    )


@pytest.mark.fast
def test_memory_saver_skips_mlx_cache_setup_when_cache_limit_is_none():
    model = _EncoderModel()

    with (
        patch("mflux.callbacks.instances.memory_saver.mx.set_cache_limit") as mock_set_cache_limit,
        patch("mflux.callbacks.instances.memory_saver.mx.clear_cache") as mock_clear_cache,
        patch("mflux.callbacks.instances.memory_saver.mx.reset_peak_memory") as mock_reset_peak_memory,
    ):
        MemorySaver(model=model, cache_limit_bytes=None)

    mock_set_cache_limit.assert_not_called()
    mock_clear_cache.assert_not_called()
    mock_reset_peak_memory.assert_not_called()


@pytest.mark.fast
def test_call_before_loop_evicts_text_encoder_for_single_seed():
    model = _EncoderModel()
    saver = MemorySaver(model=model, cache_limit_bytes=None, num_seeds=1)

    saver.call_before_loop(seed=1, prompt="a cat", latents=None, config=_config())

    assert model.text_encoder is None


@pytest.mark.fast
def test_call_before_loop_keeps_text_encoder_for_multi_seed_without_prompt_cache():
    model = _EncoderModel()
    saver = MemorySaver(model=model, cache_limit_bytes=None, num_seeds=3)

    saver.call_before_loop(seed=1, prompt="a cat", latents=None, config=_config())

    assert model.text_encoder is not None


@pytest.mark.fast
def test_call_before_loop_evicts_text_encoder_for_multi_seed_with_cached_prompt():
    model = _EncoderModel(prompt_cache={"a cat": object()})
    saver = MemorySaver(model=model, cache_limit_bytes=None, num_seeds=3)

    saver.call_before_loop(seed=2, prompt="a cat", latents=None, config=_config())

    assert model.text_encoder is None


@pytest.mark.fast
def test_call_after_loop_clears_cache_when_keep_transformer_true():
    model = _EncoderModel()
    saver = MemorySaver(model=model, keep_transformer=True, cache_limit_bytes=None)

    with (
        patch("mflux.callbacks.instances.memory_saver.gc.collect") as mock_gc_collect,
        patch("mflux.callbacks.instances.memory_saver.mx.clear_cache") as mock_clear_cache,
    ):
        saver.call_after_loop(seed=1, prompt="a cat", latents=None, config=_config())

    mock_gc_collect.assert_called_once()
    mock_clear_cache.assert_called_once()
    assert model.transformer is not None
