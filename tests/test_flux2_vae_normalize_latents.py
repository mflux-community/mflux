import mlx.core as mx
import numpy as np
import pytest

from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

pytestmark = pytest.mark.fast


def test_normalize_latents_is_the_inverse_of_unpacking_packed_latents():
    vae = Flux2VAE()
    vae.bn.running_mean = mx.random.normal((128,), key=mx.random.key(0))
    vae.bn.running_var = mx.random.uniform(0.5, 2.0, (128,), key=mx.random.key(1))
    latents = mx.random.normal((1, 32, 8, 12), key=mx.random.key(2))
    restored = vae.unpack_packed_latents(Flux2LatentCreator.patchify_latents(vae.normalize_latents(latents)))
    np.testing.assert_allclose(np.array(restored), np.array(latents), atol=1e-5)


def test_normalize_latents_applies_the_statistics_per_patchified_channel():
    # Channel c of the plain latent at offset (dy, dx) is packed channel 4c + 2dy + dx
    vae = Flux2VAE()
    vae.bn.running_mean = mx.arange(128, dtype=mx.float32)
    vae.bn.running_var = mx.ones((128,)) - vae.bn.eps
    normalized = np.array(vae.normalize_latents(mx.zeros((1, 32, 2, 2))))
    assert normalized[0, 3, 1, 0] == -(4 * 3 + 2 * 1 + 0)
