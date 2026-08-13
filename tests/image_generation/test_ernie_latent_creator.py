import mlx.core as mx
import pytest

from mflux.models.ernie_image.latent_creator.ernie_latent_creator import ErnieLatentCreator


@pytest.mark.fast
def test_pack_latents_accepts_tiled_vae_latents() -> None:
    latents = mx.zeros((1, 32, 1, 64, 64))

    packed = ErnieLatentCreator.pack_latents(latents, height=1024, width=1024)

    assert packed.shape == (1, 128, 32, 32)
