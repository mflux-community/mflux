import pytest

from mflux.models.common.training.runner import TrainingRunner
from mflux.models.common.vae.tiling_config import TilingConfig


class _OptOutVAE:
    supports_implicit_tiling = False


class _PlainVAE:
    pass


class _Model:
    def __init__(self, vae, tiling_config: TilingConfig | None = None) -> None:
        self.vae = vae
        self.tiling_config = tiling_config


class _ModelWithoutTilingConfig:
    def __init__(self) -> None:
        self.vae = _PlainVAE()


@pytest.mark.fast
def test_low_ram_training_does_not_tile_a_vae_that_opts_out():
    model = _Model(vae=_OptOutVAE())

    TrainingRunner._apply_low_ram_tiling(model)

    assert model.tiling_config is None


@pytest.mark.fast
def test_low_ram_training_tiles_a_vae_that_does_not_opt_out():
    model = _Model(vae=_PlainVAE())

    TrainingRunner._apply_low_ram_tiling(model)

    assert model.tiling_config is not None


@pytest.mark.fast
def test_training_never_replaces_a_preexisting_tiling_config():
    preset = TilingConfig(vae_decode_tiles_per_dim=None)
    model = _Model(vae=_OptOutVAE(), tiling_config=preset)

    TrainingRunner._apply_low_ram_tiling(model)

    assert model.tiling_config is preset


@pytest.mark.fast
def test_low_ram_training_ignores_a_model_with_no_tiling_config_attribute():
    model = _ModelWithoutTilingConfig()

    TrainingRunner._apply_low_ram_tiling(model)

    assert not hasattr(model, "tiling_config")


@pytest.mark.fast
def test_may_tile_implicitly_reads_the_models_vae():
    assert TilingConfig.may_tile_implicitly(_Model(vae=_OptOutVAE())) is False
    assert TilingConfig.may_tile_implicitly(_Model(vae=_PlainVAE())) is True
    assert TilingConfig.may_tile_implicitly(_Model(vae=None)) is True
