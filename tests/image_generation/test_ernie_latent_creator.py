import mlx.core as mx

from mflux.models.ernie_image.latent_creator.ernie_latent_creator import ErnieLatentCreator


class TestErnieLatentCreator:
    def test_pack_latents_handles_5d_tiled_encode_output(self):
        # Tiled VAE encoding returns 5D (B, C, 1, H_lat, W_lat) latent.
        # For a 1264x1792 image the Flux2VAE (scale 8) latent is (1, 32, 224, 158).
        latents = mx.random.normal(shape=(1, 32, 1, 224, 158))

        packed = ErnieLatentCreator.pack_latents(latents, height=1792, width=1264)

        assert packed.shape == (1, 128, 112, 79)

    def test_pack_latents_accepts_4d_latent(self):
        latents = mx.random.normal(shape=(1, 32, 224, 158))

        packed = ErnieLatentCreator.pack_latents(latents, height=1792, width=1264)

        assert packed.shape == (1, 128, 112, 79)
