from __future__ import annotations

import datetime
import json
from pathlib import Path

from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.zip_util import ZipUtil


class Statistics:
    def __init__(self):
        # Per-step training loss: the value the train step already computed, recorded free.
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.times: list[datetime.datetime] = []
        # Batch metric: an extra forward over up to 10 training samples at plot_frequency
        # ticks. Smoother, but each point costs real time (#671).
        self.batch_steps: list[int] = []
        self.batch_losses: list[float] = []
        self.batch_times: list[datetime.datetime] = []

    @staticmethod
    def from_spec(training_spec: TrainingSpec) -> "Statistics":
        if training_spec.statistics is None or training_spec.statistics.state_path is None:
            return Statistics()

        stats = Statistics()
        data = ZipUtil.unzip(
            zip_path=training_spec.checkpoint_path,
            filename=training_spec.statistics.state_path,
            loader=lambda x: json.load(open(x, "r")),
        )
        if isinstance(data, list):
            # Pre-two-series checkpoint: its single series was the per-tick batch metric.
            Statistics._load_entries(data, stats.batch_steps, stats.batch_losses, stats.batch_times)
        else:
            Statistics._load_entries(data.get("step_loss", []), stats.steps, stats.losses, stats.times)
            Statistics._load_entries(data.get("batch_loss", []), stats.batch_steps, stats.batch_losses, stats.batch_times)  # fmt: off

        return stats

    @staticmethod
    def _load_entries(entries, steps: list[int], losses: list[float], times: list[datetime.datetime]) -> None:
        for entry in entries:
            steps.append(entry["step"])
            losses.append(entry["loss"])
            times.append(datetime.datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S"))

    def append_values(self, step: int, loss: float) -> None:
        self.steps.append(step)
        self.losses.append(loss)
        self.times.append(datetime.datetime.now())

    def append_batch_values(self, step: int, loss: float) -> None:
        self.batch_steps.append(step)
        self.batch_losses.append(loss)
        self.batch_times.append(datetime.datetime.now())

    @staticmethod
    def _entries(steps: list[int], losses: list[float], times: list[datetime.datetime]) -> list[dict]:
        return [
            {"step": step, "loss": float(loss), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            for step, loss, time in zip(steps, losses, times)
        ]

    def save(self, path: Path) -> None:
        document = {
            "step_loss": Statistics._entries(self.steps, self.losses, self.times),
            "batch_loss": Statistics._entries(self.batch_steps, self.batch_losses, self.batch_times),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(document, file, indent=4)
