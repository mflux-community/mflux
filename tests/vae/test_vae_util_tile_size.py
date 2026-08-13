from unittest.mock import patch

import mlx.core as mx
import pytest

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil


class _FakeVAE:
    spatial_scale = 8

    def decode(self, latent: mx.array) -> mx.array:
        return latent


@pytest.mark.fast
def test_decode_passes_configured_tile_size_to_tiler():
    latent = mx.zeros((1, 16, 32, 32))
    with patch("mflux.models.common.vae.vae_util.VAETiler.decode_image_tiled") as mock_tiled:
        VAEUtil.decode(vae=_FakeVAE(), latent=latent, tiling_config=TilingConfig(vae_decode_tile_size=256))
    assert mock_tiled.call_args.kwargs["tile_size"] == (256, 256)


@pytest.mark.fast
def test_decode_defaults_to_512_tile_size():
    latent = mx.zeros((1, 16, 32, 32))
    with patch("mflux.models.common.vae.vae_util.VAETiler.decode_image_tiled") as mock_tiled:
        VAEUtil.decode(vae=_FakeVAE(), latent=latent, tiling_config=TilingConfig())
    assert mock_tiled.call_args.kwargs["tile_size"] == (512, 512)
