import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.variants.controlnet.z_image_turbo_controlnet import ZImageTurboControlnet
from mflux.models.z_image.variants.z_image import ZImage
from tests.cli.helpers.command_fakes import FakeModel


class FakeZImage(FakeModel):
    real = ZImage


class FakeControlnet(FakeModel):
    real = ZImageTurboControlnet


@pytest.mark.fast
def test_the_fake_rejects_a_constructor_kwarg_the_real_model_does_not_take():
    with pytest.raises(TypeError):
        FakeZImage(bogus=1)


@pytest.mark.fast
def test_the_fake_rejects_a_generate_kwarg_the_real_model_does_not_take():
    with pytest.raises(TypeError):
        FakeZImage().generate_image(seed=1, prompt="p", bogus=1)


@pytest.mark.fast
def test_the_fake_rejects_a_generate_call_missing_a_required_kwarg():
    with pytest.raises(TypeError):
        FakeControlnet().generate_image(seed=1, prompt="p")


@pytest.mark.fast
def test_the_fake_holds_the_real_default_config_when_none_is_passed():
    # ZImage defaults model_config to Turbo; a load() that drops the kwarg must not read as None.
    assert FakeZImage().model_config is ModelConfig.z_image_turbo()
