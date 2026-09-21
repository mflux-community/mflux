from argparse import Namespace

import pytest

from mflux.callbacks.callback_manager import CallbackManager
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux.model.flux_vae.vae import VAE as Flux1VAE
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE


class _OptOutVAE:
    supports_implicit_tiling = False


class _PlainVAE:
    pass


class _Model:
    def __init__(self, vae, tiling_config: TilingConfig | None = None) -> None:
        self.vae = vae
        self.tiling_config = tiling_config


class _Args:
    @staticmethod
    def low_ram() -> Namespace:
        return Namespace(
            low_ram=True,
            mlx_cache_limit_gb=None,
            vae_tiling=False,
            vae_tile_size=None,
            seed=[],
            image_path=[],
        )


@pytest.mark.fast
def test_low_ram_does_not_tile_a_vae_that_opts_out_of_implicit_tiling():
    model = _Model(vae=_OptOutVAE())

    MemorySaver(model=model, cache_limit_bytes=1000**3)

    assert model.tiling_config is None


@pytest.mark.fast
def test_low_ram_still_tiles_a_vae_that_does_not_opt_out():
    model = _Model(vae=_PlainVAE())

    MemorySaver(model=model, cache_limit_bytes=1000**3)

    assert model.tiling_config is not None


@pytest.mark.fast
def test_a_preexisting_tiling_config_is_never_replaced():
    explicit = TilingConfig(vae_decode_tile_size=384)
    model = _Model(vae=_OptOutVAE(), tiling_config=explicit)

    MemorySaver(model=model, cache_limit_bytes=1000**3)

    assert model.tiling_config is explicit


@pytest.mark.fast
def test_low_ram_tiles_a_model_that_exposes_no_vae():
    model = _Model(vae=None)

    MemorySaver(model=model, cache_limit_bytes=1000**3)

    assert model.tiling_config is not None


@pytest.mark.fast
def test_flux2_vae_opts_out_of_implicit_tiling():
    assert Flux2VAE.supports_implicit_tiling is False


@pytest.mark.fast
@pytest.mark.parametrize("vae_class", [QwenVAE, Flux1VAE])
def test_other_vaes_keep_implicit_tiling(vae_class):
    assert getattr(vae_class, "supports_implicit_tiling", True) is True


@pytest.mark.fast
def test_low_ram_leaves_a_real_flux2_klein_untiled():
    model = Flux2Klein.__new__(Flux2Klein)
    model.callbacks = CallbackRegistry()
    model.vae = Flux2VAE()
    model.tiling_config = None

    CallbackManager._register_memory_saver(_Args.low_ram(), model)

    assert model.tiling_config is None


@pytest.mark.fast
def test_low_ram_tiles_a_real_krea2():
    model = Krea2.__new__(Krea2)
    model.callbacks = CallbackRegistry()
    model.vae = QwenVAE()
    model.tiling_config = None

    CallbackManager._register_memory_saver(_Args.low_ram(), model)

    assert model.tiling_config is not None


@pytest.mark.fast
def test_explicit_vae_tiling_still_tiles_a_real_flux2_klein():
    model = Flux2Klein.__new__(Flux2Klein)
    model.callbacks = CallbackRegistry()
    model.vae = Flux2VAE()
    model.tiling_config = None
    args = _Args.low_ram()
    args.vae_tiling = True

    CallbackManager._register_memory_saver(args, model)

    assert model.tiling_config is not None
    assert model.tiling_config.vae_decode_tiles_per_dim > 1
