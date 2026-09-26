import mlx.core as mx
import pytest

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers import SCHEDULER_REGISTRY
from mflux.models.qwen21.model.qwen21_scheduler import ViggleTurboScheduler


@pytest.fixture
def turbo_config():
    return Config(
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=6,
        width=1024,
        height=1024,
        scheduler="viggle_turbo",
    )


@pytest.mark.fast
def test_registry_exposes_viggle_turbo():
    assert SCHEDULER_REGISTRY["viggle_turbo"] is ViggleTurboScheduler
    assert SCHEDULER_REGISTRY["ViggleTurboScheduler"] is ViggleTurboScheduler


@pytest.mark.fast
def test_raw_nodes_are_shifted_like_default_nodes(turbo_config):
    # The model card ships RAW sigma nodes; the resolution-dependent flow shift must be
    # applied exactly as for the default schedule (shift moves mid/low nodes up).
    scheduler = turbo_config.scheduler
    sigmas = [float(s) for s in scheduler.sigmas]
    assert len(sigmas) == 7 and sigmas[-1] == 0.0
    assert sigmas[0] == 1.0
    raw = list(ViggleTurboScheduler.SIGMA_NODES)
    for shifted, node in zip(sigmas[1:-1], raw[1:]):
        assert shifted > node  # shift pulls every non-1.0 node toward higher noise
    linear = Config(
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=6,
        width=1024,
        height=1024,
        scheduler="linear",
    ).scheduler
    assert sigmas != [float(s) for s in linear.sigmas]


@pytest.mark.fast
def test_timestep_routing_lands_on_sigma_nodes(turbo_config):
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    sigmas = [float(s) for s in turbo_config.scheduler.sigmas]
    for t in range(6):
        routed = float(Qwen21Transformer._compute_timestep(t, turbo_config)[0])
        assert routed == pytest.approx(sigmas[t])


@pytest.mark.fast
def test_euler_step_between_nodes(turbo_config):
    scheduler = turbo_config.scheduler
    latents = mx.ones((1, 4))
    noise = mx.full((1, 4), -0.5)
    dt = float(scheduler.sigmas[1] - scheduler.sigmas[0])
    out = scheduler.step(noise, 0, latents)
    assert float(out[0][0]) == pytest.approx(1.0 - 0.5 * dt)


@pytest.mark.fast
def test_rejects_non_six_step_counts():
    config = Config(
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=40,
        width=1024,
        height=1024,
    )
    with pytest.raises(ValueError, match="6 trained sigma nodes"):
        ViggleTurboScheduler(config)
