from types import SimpleNamespace

import mlx.core as mx
import pytest

from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.ideogram4.latent_creator.ideogram4_latent_creator import Ideogram4LatentCreator

SIZE = 256


def _decoded(height: int, width: int) -> mx.array:
    return mx.zeros((1, 3, height, width))


class RecordingPackedVAE:
    """Flux2-style VAE: decode_packed_latents does BN denorm + unpatchify, so it only
    accepts the 4x-patchified channel count and rejects VAE-ready latents."""

    latent_channels = 32

    def __init__(self):
        self.calls = []

    def decode(self, latents: mx.array) -> mx.array:
        self.calls.append("decode")
        if latents.shape[1] != self.latent_channels:
            raise ValueError(f"decode expects {self.latent_channels} channels, got {latents.shape[1]}")
        return _decoded(SIZE, SIZE)

    def decode_packed_latents(self, latents: mx.array) -> mx.array:
        self.calls.append("decode_packed_latents")
        if latents.shape[1] != 4 * self.latent_channels:
            raise ValueError(
                f"decode_packed_latents expects {4 * self.latent_channels} channels, got {latents.shape[1]}"
            )
        return _decoded(SIZE, SIZE)


class RecordingPlainVAE:
    """Flux-style VAE with no decode_packed_latents at all."""

    latent_channels = 16

    def __init__(self):
        self.calls = []

    def decode(self, latents: mx.array) -> mx.array:
        self.calls.append("decode")
        return _decoded(SIZE, SIZE)


def _handler(tmp_path, vae, latent_creator) -> StepwiseHandler:
    model = SimpleNamespace(vae=vae, bits=None, lora_paths=None, lora_scales=None)
    return StepwiseHandler(model=model, output_dir=str(tmp_path), latent_creator=latent_creator)


def _config(model_config: ModelConfig) -> Config:
    return Config(model_config=model_config, num_inference_steps=1, height=SIZE, width=SIZE)


@pytest.mark.fast
def test_stepwise_ideogram4_uses_plain_decode(tmp_path) -> None:
    # Ideogram4's unpack_latents applies shift/scale and returns VAE-ready 32ch latents.
    vae = RecordingPackedVAE()
    handler = _handler(tmp_path, vae, Ideogram4LatentCreator)
    latents = mx.zeros((1, (SIZE // 16) * (SIZE // 16), 128))

    assert Ideogram4LatentCreator.unpack_latents(latents, SIZE, SIZE).shape[1] == vae.latent_channels

    handler.call_before_loop(seed=1, prompt="a", latents=latents, config=_config(ModelConfig.ideogram4_fp8()))

    assert vae.calls == ["decode"]
    assert (tmp_path / "seed_1_step0of1.png").exists()


@pytest.mark.fast
def test_stepwise_flux2_uses_packed_decode(tmp_path) -> None:
    # Flux2's unpack_latents returns 128ch patchified latents that still need BN denorm.
    vae = RecordingPackedVAE()
    handler = _handler(tmp_path, vae, Flux2LatentCreator)
    latents = mx.zeros((1, (SIZE // 16) * (SIZE // 16), 128))

    assert Flux2LatentCreator.unpack_latents(latents, SIZE, SIZE).shape[1] == 4 * vae.latent_channels

    handler.call_before_loop(seed=1, prompt="a", latents=latents, config=_config(ModelConfig.flux2_klein_4b()))

    assert vae.calls == ["decode_packed_latents"]
    assert (tmp_path / "seed_1_step0of1.png").exists()


@pytest.mark.fast
def test_stepwise_vae_without_packed_decode_falls_back(tmp_path) -> None:
    vae = RecordingPlainVAE()
    handler = _handler(tmp_path, vae, FluxLatentCreator)
    latents = mx.zeros((1, (SIZE // 16) * (SIZE // 16), 64))

    handler.call_before_loop(seed=1, prompt="a", latents=latents, config=_config(ModelConfig.dev()))

    assert vae.calls == ["decode"]
