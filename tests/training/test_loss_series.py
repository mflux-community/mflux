import json

import pytest

from mflux.models.common.training.state.training_spec import MonitoringSpec
from mflux.models.common.training.statistics.statistics import Statistics


@pytest.mark.fast
def test_statistics_saves_both_series(tmp_path):
    stats = Statistics()
    stats.append_values(step=1, loss=0.5)
    stats.append_values(step=2, loss=0.4)
    stats.append_batch_values(step=20, loss=0.45)

    path = tmp_path / "loss.json"
    stats.save(path)
    document = json.load(open(path))

    assert [e["step"] for e in document["step_loss"]] == [1, 2]
    assert [e["loss"] for e in document["step_loss"]] == [0.5, 0.4]
    assert [e["step"] for e in document["batch_loss"]] == [20]


@pytest.mark.fast
def test_statistics_loads_both_series(monkeypatch, tmp_path):
    stats = Statistics()
    stats.append_values(step=1, loss=0.5)
    stats.append_batch_values(step=20, loss=0.45)
    path = tmp_path / "loss.json"
    stats.save(path)

    document = json.load(open(path))
    reloaded = Statistics()
    Statistics._load_entries(document["step_loss"], reloaded.steps, reloaded.losses, reloaded.times)
    Statistics._load_entries(document["batch_loss"], reloaded.batch_steps, reloaded.batch_losses, reloaded.batch_times)

    assert reloaded.steps == [1] and reloaded.losses == [0.5]
    assert reloaded.batch_steps == [20] and reloaded.batch_losses == [0.45]


@pytest.mark.fast
def test_legacy_single_series_checkpoint_loads_as_batch_metric(monkeypatch):
    # Pre-two-series checkpoints hold one flat list, and those entries were the
    # per-tick batch metric, so a resumed run must not graft them onto the new
    # per-step series.
    from types import SimpleNamespace

    from mflux.models.common.training.statistics import statistics as statistics_module

    legacy = [{"step": 10, "loss": 0.6, "time": "2026-08-01 10:00:00"}]
    monkeypatch.setattr(statistics_module.ZipUtil, "unzip", lambda **kwargs: legacy)
    fake_spec = SimpleNamespace(
        statistics=SimpleNamespace(state_path="loss.json"),
        checkpoint_path="checkpoint.zip",
    )

    stats = Statistics.from_spec(fake_spec)

    assert stats.batch_steps == [10]
    assert stats.batch_losses == [0.6]
    assert stats.steps == []


@pytest.mark.fast
def test_plot_frequency_is_optional_and_defaults_to_20():
    spec = MonitoringSpec.create(
        {"generate_image_frequency": 15},
        preview_prompts=["a prompt"],
        preview_prompt_names=["01"],
    )
    assert spec.plot_frequency == 20


@pytest.mark.fast
def test_explicit_plot_frequency_still_wins():
    spec = MonitoringSpec.create(
        {"plot_frequency": 3, "generate_image_frequency": 15},
        preview_prompts=["a prompt"],
        preview_prompt_names=["01"],
    )
    assert spec.plot_frequency == 3
