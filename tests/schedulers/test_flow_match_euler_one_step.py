from types import SimpleNamespace

import pytest

from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler


def _scheduler(num_steps: int) -> FlowMatchEulerDiscreteScheduler:
    config = SimpleNamespace(
        num_inference_steps=num_steps,
        model_config=SimpleNamespace(),
    )
    return FlowMatchEulerDiscreteScheduler(config)


@pytest.mark.fast
def test_one_step_schedule_is_full_denoise():
    # Crashed with ZeroDivisionError before the fix (#494). The 1-step schedule
    # must match what get_timesteps_and_sigmas and the set_mu path already produce.
    scheduler = _scheduler(1)
    assert scheduler.sigmas.tolist() == [1.0, 0.0]
    assert scheduler.timesteps.tolist() == [1000.0]


# The full schedules the fix must leave alone, pinned rather than described: a test named
# "unchanged" that only checks the endpoints would pass while the interior moved.
EXPECTED_SCHEDULES = {
    2: ([1.0, 0.02, 0.0], [1000.0, 20.0]),
    4: ([1.0, 0.847524, 0.584184, 0.02, 0.0], [1000.0, 847.5239, 584.1836, 20.0]),
}


@pytest.mark.fast
@pytest.mark.parametrize("num_steps", [2, 4])
def test_multi_step_schedules_are_unchanged(num_steps):
    expected_sigmas, expected_timesteps = EXPECTED_SCHEDULES[num_steps]
    scheduler = _scheduler(num_steps)

    assert scheduler.sigmas.tolist() == pytest.approx(expected_sigmas, abs=1e-5)
    assert scheduler.timesteps.tolist() == pytest.approx(expected_timesteps, abs=1e-3)


@pytest.mark.fast
def test_one_step_matches_sibling_paths():
    timesteps, sigmas = FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(
        image_seq_len=1024,
        num_inference_steps=1,
    )
    scheduler = _scheduler(1)
    assert scheduler.sigmas.tolist() == sigmas.tolist()
    assert scheduler.timesteps.tolist() == timesteps.tolist()
