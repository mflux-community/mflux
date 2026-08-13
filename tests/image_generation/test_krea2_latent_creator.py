import mlx.core as mx
import pytest

from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator


@pytest.mark.fast
def test_pack_latents_accepts_tiled_vae_latents() -> None:
    latents = mx.zeros((1, 16, 1, 128, 128))

    packed = Krea2LatentCreator.pack_latents(latents, height=1024, width=1024)

    assert packed.shape == (1, 16, 128, 128)


@pytest.mark.fast
def test_pack_latents_passes_through_untiled_latents() -> None:
    latents = mx.zeros((1, 16, 128, 128))

    packed = Krea2LatentCreator.pack_latents(latents, height=1024, width=1024)

    assert packed.shape == (1, 16, 128, 128)
