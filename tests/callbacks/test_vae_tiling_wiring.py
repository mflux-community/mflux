from argparse import Namespace
from dataclasses import replace
from unittest.mock import patch

import pytest

from mflux.callbacks.callback_manager import CallbackManager
from mflux.models.common.vae.tiling_config import TilingConfig


class _Callbacks:
    def register(self, callback) -> None:
        pass


class _Model:
    def __init__(self, tiling_config: TilingConfig | None = None) -> None:
        self.tiling_config = tiling_config
        self.callbacks = _Callbacks()


def _args(**overrides) -> Namespace:
    defaults = {"vae_tiling": False, "vae_tile_size": None, "low_ram": False, "mlx_cache_limit_gb": None}
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.mark.fast
def test_no_flags_leaves_tiling_config_untouched():
    model = _Model()
    CallbackManager._apply_vae_tiling(_args(), model)
    assert model.tiling_config is None


@pytest.mark.fast
def test_vae_tiling_flag_installs_default_config():
    model = _Model()
    CallbackManager._apply_vae_tiling(_args(vae_tiling=True), model)
    assert model.tiling_config is not None
    assert model.tiling_config.vae_decode_tile_size == 512


@pytest.mark.fast
def test_vae_tile_size_alone_implies_tiling():
    model = _Model()
    CallbackManager._apply_vae_tiling(_args(vae_tile_size=256), model)
    assert model.tiling_config is not None
    assert model.tiling_config.vae_decode_tile_size == 256


@pytest.mark.fast
def test_tile_size_overrides_preexisting_config_without_replacing_other_fields():
    preexisting = replace(TilingConfig(), vae_decode_overlap=16)
    model = _Model(tiling_config=preexisting)
    CallbackManager._apply_vae_tiling(_args(vae_tiling=True, vae_tile_size=256), model)
    assert model.tiling_config.vae_decode_tile_size == 256
    assert model.tiling_config.vae_decode_overlap == 16


@pytest.mark.fast
def test_vae_tiling_alone_keeps_preexisting_config():
    preexisting = TilingConfig(vae_decode_tile_size=384)
    model = _Model(tiling_config=preexisting)
    CallbackManager._apply_vae_tiling(_args(vae_tiling=True), model)
    assert model.tiling_config is preexisting


@pytest.mark.fast
def test_explicit_tile_size_wins_over_low_ram_default():
    model = _Model()
    with (
        patch("mflux.callbacks.instances.memory_saver.mx.set_cache_limit"),
        patch("mflux.callbacks.instances.memory_saver.mx.clear_cache"),
        patch("mflux.callbacks.instances.memory_saver.mx.reset_peak_memory"),
    ):
        CallbackManager._register_memory_saver(_args(low_ram=True, vae_tile_size=256), model)
    assert model.tiling_config.vae_decode_tile_size == 256
