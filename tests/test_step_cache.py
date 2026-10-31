import mlx.core as mx
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.step_cache.step_cache import StepCache
from mflux.models.qwen21.model.qwen21_transformer.qwen21_time_text_embed import Qwen21TimeTextEmbed


def _config(num_steps: int, image_strength: float | None = None) -> Config:
    return Config(
        width=512,
        height=512,
        guidance=1.0,
        scheduler="linear",
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=num_steps,
        image_path="init.png" if image_strength is not None else None,
        image_strength=image_strength,
    )


@pytest.mark.fast
class TestStepCacheSelection:
    def test_inactive_without_ratio(self):
        cache = StepCache.for_run(_config(40), ratio=None)
        assert not cache.active
        assert cache.reuse(10) is None

    def test_sigma_signal_skips_ratio_of_steps_inside_protected_window(self):
        cache = StepCache.for_run(_config(40), ratio=0.25)
        assert len(cache.skip_steps) == 10
        assert all(4 <= t < 36 for t in cache.skip_steps)  # first/last 10% always run

    def test_custom_signal_fn_drives_selection(self):
        embed = Qwen21TimeTextEmbed(embedding_dim=64)
        cache = StepCache.for_run(_config(40), ratio=0.25, signal_fn=embed)
        assert len(cache.skip_steps) == 10
        assert all(4 <= t < 36 for t in cache.skip_steps)

    def test_signal_fn_receives_one_sigma_per_run_step(self):
        seen = {}

        def signal_fn(step_sigmas):
            seen["sigmas"] = step_sigmas
            return step_sigmas

        config = _config(40)
        StepCache.for_run(config, ratio=0.25, signal_fn=signal_fn)
        expected = config.scheduler.sigmas[:40]
        assert seen["sigmas"].shape == (40,)
        assert mx.allclose(seen["sigmas"], expected.astype(mx.float32))

    def test_img2img_window_follows_the_run_start(self):
        config = _config(40, image_strength=0.6)
        first = config.init_time_step
        cache = StepCache.for_run(config, ratio=0.25)
        run_length = 40 - first
        protected = int(run_length * StepCache.PROTECTED_FRACTION)
        assert cache.skip_steps
        assert all(first + max(protected, 1) <= t < 40 - protected for t in cache.skip_steps)

    def test_selection_does_not_consume_the_progress_bar(self):
        config = _config(40)
        StepCache.for_run(config, ratio=0.25)
        assert not config.time_steps.disable  # the loop's tqdm bar is still live
        assert sum(1 for _ in config.time_steps) == 40

    def test_selection_is_deterministic(self):
        first = StepCache.for_run(_config(40), ratio=0.3).skip_steps
        second = StepCache.for_run(_config(40), ratio=0.3).skip_steps
        assert first == second

    def test_short_runs_skip_nothing(self):
        assert not StepCache.for_run(_config(StepCache.MIN_STEPS - 1), ratio=0.3).active

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_ratio_out_of_range_raises(self, ratio):
        with pytest.raises(ValueError):
            StepCache.for_run(_config(40), ratio=ratio)


@pytest.mark.fast
class TestStepCacheReuse:
    def test_reuses_last_stored_output_only_on_skip_steps(self):
        cache = StepCache(frozenset({5}))
        first = mx.ones((1, 4))
        cache.store(first)
        assert cache.reuse(4) is None
        assert cache.reuse(5) is first

    def test_skip_step_before_any_computed_step_runs_the_model(self):
        cache = StepCache(frozenset({0}))
        assert cache.reuse(0) is None

    def test_generation_parameters_only_record_reuse_runs(self):
        assert StepCache.generation_parameters(None) == {}
        assert StepCache.generation_parameters(0.25) == {"step_cache_ratio": 0.25}


@pytest.mark.fast
class TestStepCacheProtectedWindow:
    @pytest.mark.parametrize("num_steps", [25, 50])
    def test_larger_window_protects_early_structure_steps(self, num_steps):
        config = _config(num_steps)
        cache = StepCache.for_run(config, ratio=0.3, protected_fraction=0.32)
        protected = int(num_steps * 0.32)
        assert len(cache.skip_steps) == round(num_steps * 0.3)
        assert all(protected <= t < num_steps - protected for t in cache.skip_steps)

    def test_window_is_relative_to_img2img_run(self):
        config = _config(40, image_strength=0.6)
        cache = StepCache.for_run(config, ratio=0.3, protected_fraction=0.32)
        protected = int((40 - config.init_time_step) * 0.32)
        assert cache.skip_steps
        assert all(config.init_time_step + protected <= t < 40 - protected for t in cache.skip_steps)

    @pytest.mark.parametrize("num_steps", [25, 40])
    def test_fully_protected_run_skips_nothing(self, num_steps):
        assert not StepCache.for_run(_config(num_steps), ratio=0.3, protected_fraction=0.5).active

    @pytest.mark.parametrize("fraction", [-0.1, 0.51, float("nan"), float("inf")])
    def test_invalid_window_raises(self, fraction):
        with pytest.raises(ValueError, match="protected fraction"):
            StepCache.for_run(_config(40), ratio=0.3, protected_fraction=fraction)

    def test_near_integer_window_is_not_shortened(self):
        cache = StepCache.for_run(_config(100), ratio=0.9, protected_fraction=0.29)
        assert cache.skip_steps == frozenset(range(29, 71))

    def test_default_reads_class_constant_at_call_time(self, monkeypatch):
        monkeypatch.setattr(StepCache, "PROTECTED_FRACTION", 0.3)
        config = _config(40)
        cache = StepCache.for_run(config, ratio=0.9)
        direct = StepCache.select_skip_steps(list(range(40)), config.scheduler.sigmas, 0.9)
        assert cache.skip_steps == direct == frozenset(range(12, 28))
        explicit = StepCache.for_run(config, ratio=0.9, protected_fraction=0.1)
        assert explicit.skip_steps == frozenset(range(4, 36))

    @pytest.mark.parametrize("fraction", [-0.1, 0.51, float("nan"), float("inf")])
    def test_invalid_window_raises_with_disabled_cache(self, fraction):
        with pytest.raises(ValueError, match="protected fraction"):
            StepCache.for_run(_config(40), ratio=None, protected_fraction=fraction)

    def test_disabled_cache_accepts_valid_window(self):
        assert not StepCache.for_run(_config(40), ratio=None, protected_fraction=0.5).active
