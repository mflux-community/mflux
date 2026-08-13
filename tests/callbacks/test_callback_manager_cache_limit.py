from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mflux.callbacks.callback_manager import CallbackManager


@pytest.mark.fast
def test_register_memory_saver_sets_mlx_cache_limit_without_low_ram():
    args = Namespace(low_ram=False, mlx_cache_limit_gb=2.5)
    model = SimpleNamespace(callbacks=MagicMock())

    with (
        patch("mflux.callbacks.callback_manager.mx.set_cache_limit") as mock_set_cache_limit,
        patch("mflux.callbacks.callback_manager.mx.clear_cache") as mock_clear_cache,
        patch("mflux.callbacks.callback_manager.mx.reset_peak_memory") as mock_reset_peak_memory,
    ):
        memory_saver = CallbackManager._register_memory_saver(args=args, model=model)

    # MemorySaver is always registered when a cache limit is set without low_ram.
    assert memory_saver is not None
    mock_set_cache_limit.assert_called_once_with(int(2.5 * (1000**3)))
    mock_clear_cache.assert_called_once()
    mock_reset_peak_memory.assert_called_once()
    model.callbacks.register.assert_called_once()


@pytest.mark.fast
def test_register_memory_saver_registers_by_default_without_low_ram_or_cache_limit():
    args = Namespace(low_ram=False, seed=[1, 2, 3])
    model = SimpleNamespace(callbacks=MagicMock())

    memory_saver = CallbackManager._register_memory_saver(args=args, model=model)

    assert memory_saver is not None
    assert memory_saver._num_seeds == 3
    assert memory_saver.keep_transformer is True
    model.callbacks.register.assert_called_once_with(memory_saver)


@pytest.mark.fast
def test_register_memory_saver_uses_mlx_cache_limit_for_low_ram_mode():
    args = Namespace(low_ram=True, mlx_cache_limit_gb=3.0, seed=[42, 43], image_path=None)
    model = SimpleNamespace(callbacks=MagicMock())
    mocked_memory_saver = object()

    with patch("mflux.callbacks.callback_manager.MemorySaver", return_value=mocked_memory_saver) as mock_memory_saver:
        memory_saver = CallbackManager._register_memory_saver(args=args, model=model)

    assert memory_saver is mocked_memory_saver
    _, kwargs = mock_memory_saver.call_args
    assert kwargs["cache_limit_bytes"] == int(3.0 * (1000**3))
    assert kwargs["num_seeds"] == 2
    assert kwargs["keep_transformer"] is True
    model.callbacks.register.assert_called_once_with(mocked_memory_saver)
