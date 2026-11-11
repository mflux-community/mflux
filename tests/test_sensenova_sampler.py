import math

import mlx.core as mx
import pytest

from mflux.models.sensenova.latent_creator.sensenova_latent_creator import SenseNovaLatentCreator
from mflux.models.sensenova.model.sensenova_scheduler.sensenova_scheduler import SenseNovaScheduler
from mflux.models.sensenova.model.sensenova_transformer.sensenova_transformer import SenseNovaTransformer


@pytest.mark.fast
class TestSenseNovaScheduler:
    def test_the_timesteps_run_from_noise_at_0_to_the_image_at_1(self):
        for steps in (1, 4, 8, 50):
            timesteps = SenseNovaScheduler(steps).timesteps
            assert len(timesteps) == steps + 1
            assert timesteps[0] == 0.0
            assert math.isclose(timesteps[-1], 1.0, abs_tol=1e-12)
            assert timesteps == sorted(timesteps)

    def test_the_shift_spends_more_steps_near_noise(self):
        # shift 3 on sigma = 1 - t: t = 0.5 maps to sigma = 3 * 0.5 / (1 + 2 * 0.5) = 0.75, so t = 0.25
        assert SenseNovaScheduler(2, shift=3.0).timesteps == pytest.approx([0.0, 0.25, 1.0])
        assert SenseNovaScheduler(2, shift=1.0).timesteps == pytest.approx([0.0, 0.5, 1.0])

    def test_a_step_moves_the_sample_along_the_velocity_by_the_time_step(self):
        scheduler = SenseNovaScheduler(2, shift=1.0)
        latents, velocity = mx.ones((1, 3, 2, 2)), mx.full((1, 3, 2, 2), 4.0)

        assert mx.allclose(scheduler.step(latents, velocity, 0), mx.full((1, 3, 2, 2), 3.0)).item()
        assert scheduler.model_time(1).tolist() == [0.5]


@pytest.mark.fast
class TestSenseNovaVelocity:
    def test_the_velocity_points_from_the_sample_to_the_prediction(self):
        image, x_pred = mx.zeros((1, 3, 2, 2)), mx.ones((1, 3, 2, 2))

        assert mx.allclose(SenseNovaTransformer.velocity(x_pred, image, 0.5), mx.full((1, 3, 2, 2), 2.0)).item()

    def test_the_divisor_is_clamped_close_to_the_end_of_the_schedule(self):
        image, x_pred = mx.zeros((1, 3, 2, 2)), mx.ones((1, 3, 2, 2))

        assert mx.allclose(SenseNovaTransformer.velocity(x_pred, image, 1.0), mx.full((1, 3, 2, 2), 50.0)).item()


@pytest.mark.fast
class TestSenseNovaLatentCreator:
    def test_the_noise_scale_follows_the_token_count_up_to_a_cap(self):
        assert SenseNovaLatentCreator.noise_scale(1024, 1024) == 4.0
        assert SenseNovaLatentCreator.noise_scale(2048, 2048) == 8.0
        assert SenseNovaLatentCreator.noise_scale(1536, 2720) == pytest.approx(math.sqrt(48 * 85 / 64))
        assert SenseNovaLatentCreator.noise_scale(8192, 8192) == 16.0

    def test_the_noise_is_an_rgb_image_scaled_by_the_noise_scale(self):
        noise = SenseNovaLatentCreator.create_noise(seed=42, height=64, width=96)

        assert noise.shape == (1, 3, 64, 96)
        assert noise.dtype == mx.float32
        unit = SenseNovaLatentCreator.create_noise(seed=42, height=64, width=96) / SenseNovaLatentCreator.noise_scale(64, 96)  # fmt: skip
        assert float(mx.std(unit)) == pytest.approx(1.0, abs=0.05)
        assert mx.array_equal(noise, SenseNovaLatentCreator.create_noise(seed=42, height=64, width=96)).item()
        assert not mx.array_equal(noise, SenseNovaLatentCreator.create_noise(seed=43, height=64, width=96)).item()

    def test_unpacking_is_the_identity_because_there_is_no_vae(self):
        latents = mx.ones((1, 3, 64, 64))

        assert SenseNovaLatentCreator.unpack_latents(latents, 64, 64) is latents

    @pytest.mark.parametrize(("height", "width"), [(1024, 1024), (1536, 2720), (32, 64), (2720, 1536)])
    def test_sizes_that_are_multiples_of_32_pass_without_a_warning(self, height, width, caplog):
        SenseNovaLatentCreator.check_size(height, width)

        assert caplog.records == []

    @pytest.mark.parametrize(("height", "width"), [(1000, 1024), (1024, 1040), (16, 32)])
    def test_other_sizes_are_an_error(self, height, width):
        with pytest.raises(ValueError, match="multiples of 32"):
            SenseNovaLatentCreator.check_size(height, width)
