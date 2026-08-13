from types import SimpleNamespace

import pytest

from mflux.models.common.training.optimization.optimizer import Optimizer
from mflux.models.common.training.state.training_spec import OptimizerSpec


def _spec(**optimizer_kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer=OptimizerSpec(name="AdamW", learning_rate=1e-4, **optimizer_kwargs),
        checkpoint_path=None,
    )


@pytest.mark.fast
def test_optimizer_params_are_forwarded():
    optimizer = Optimizer.from_spec(_spec(optimizer_params={"weight_decay": 0.123, "eps": 1e-6})).optimizer
    assert optimizer.weight_decay == 0.123
    assert optimizer.eps == 1e-6


@pytest.mark.fast
def test_optimizer_defaults_unchanged_without_params():
    optimizer = Optimizer.from_spec(_spec()).optimizer
    default_adamw_weight_decay = 0.01
    assert optimizer.weight_decay == default_adamw_weight_decay
