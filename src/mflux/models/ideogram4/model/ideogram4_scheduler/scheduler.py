import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True, slots=True)
class Ideogram4LogitNormalSchedule:
    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: float | np.ndarray) -> np.ndarray:
        values = np.asarray(t, dtype=np.float64)
        scalar = values.ndim == 0
        work = np.atleast_1d(values)
        normal = NormalDist()
        z = np.empty_like(work, dtype=np.float64)
        z[work <= 0.0] = -np.inf
        z[work >= 1.0] = np.inf
        inner = (work > 0.0) & (work < 1.0)
        if np.any(inner):
            ndtri = np.vectorize(normal.inv_cdf, otypes=[np.float64])
            z[inner] = ndtri(work[inner])
        y = self.mean + self.std * z
        shifted = 1.0 - (1.0 / (1.0 + np.exp(-y)))
        t_min = 1.0 / (1.0 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1.0 + math.exp(0.5 * self.logsnr_min))
        result = np.clip(shifted, t_min, t_max).astype(np.float32)
        return result[0] if scalar else result


@dataclass(frozen=True, slots=True)
class Ideogram4SamplerPreset:
    num_steps: int
    guidance_schedule: tuple[float, ...]
    mu: float
    std: float = 1.0

    def __post_init__(self) -> None:
        if len(self.guidance_schedule) != self.num_steps:
            raise ValueError(
                f"guidance_schedule has length {len(self.guidance_schedule)}, expected num_steps={self.num_steps}"
            )


class Ideogram4Scheduler:
    PRESETS: dict[str, Ideogram4SamplerPreset] = {
        "V4_QUALITY_48": Ideogram4SamplerPreset(
            num_steps=48,
            guidance_schedule=(3.0,) * 3 + (7.0,) * 45,
            mu=0.0,
            std=1.5,
        ),
        "V4_DEFAULT_20": Ideogram4SamplerPreset(
            num_steps=20,
            guidance_schedule=(3.0,) * 2 + (7.0,) * 18,
            mu=0.0,
            std=1.75,
        ),
        "V4_TURBO_12": Ideogram4SamplerPreset(
            num_steps=12,
            guidance_schedule=(3.0,) + (7.0,) * 11,
            mu=0.5,
            std=1.75,
        ),
    }

    @staticmethod
    def get_preset(name: str | None) -> Ideogram4SamplerPreset:
        key = (name or "V4_DEFAULT_20").strip().upper()
        try:
            return Ideogram4Scheduler.PRESETS[key]
        except KeyError as exc:
            supported = ", ".join(sorted(Ideogram4Scheduler.PRESETS))
            raise ValueError(f"Unknown Ideogram 4 preset {name!r}. Supported: {supported}") from exc

    @staticmethod
    def make_step_intervals(num_steps: int) -> np.ndarray:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        return np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)

    @staticmethod
    def make_timesteps(
        *,
        num_steps: int,
        height: int,
        width: int,
        mu: float,
        std: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        schedule = Ideogram4Scheduler._get_schedule_for_resolution((height, width), known_mean=mu, std=std)
        intervals = Ideogram4Scheduler.make_step_intervals(num_steps)
        t_values = schedule(intervals[1:])
        s_values = schedule(intervals[:-1])
        return t_values, s_values

    @staticmethod
    def _get_schedule_for_resolution(
        image_resolution: tuple[int, int],
        known_resolution: tuple[int, int] = (512, 512),
        known_mean: float = 1.0,
        std: float = 1.0,
    ) -> Ideogram4LogitNormalSchedule:
        num_pixels = image_resolution[0] * image_resolution[1]
        known_pixels = known_resolution[0] * known_resolution[1]
        mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
        return Ideogram4LogitNormalSchedule(mean=mean, std=std)
