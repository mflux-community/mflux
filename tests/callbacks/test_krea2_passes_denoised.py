import mlx.core as mx
import pytest

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.krea2.variants import Krea2


class _FakeTokenizer:
    def tokenize(self, prompt: str, images=None, max_length=None, **kwargs):
        length = max(len(prompt), 1)
        input_ids = mx.arange(length, dtype=mx.int32)[None, :]
        attention_mask = mx.ones((1, length), dtype=mx.int32)
        return type("TokenizerOutput", (), {"input_ids": input_ids, "attention_mask": attention_mask})()


class _FakeTextEncoder:
    def get_prompt_embeds(self, input_ids, attention_mask=None):
        return mx.zeros((1, input_ids.shape[1], 8), dtype=mx.float32)


class _FakeTransformer:
    def __call__(self, hidden_states, timestep, context, attention_mask=None):
        return mx.ones_like(hidden_states)


class _FakeVAE:
    def decode(self, latents):
        return mx.zeros((latents.shape[0], 3, latents.shape[2] * 8, latents.shape[3] * 8))


class _RecordingCallback:
    def __init__(self) -> None:
        self.calls: list[tuple[int, mx.array, mx.array | None, object]] = []

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, denoised=None):
        self.calls.append((t, latents, denoised, config))


class _Fixtures:
    @staticmethod
    def krea2_model() -> Krea2:
        model = Krea2.__new__(Krea2)
        model.model_config = ModelConfig.krea2()
        model.callbacks = CallbackRegistry()
        model.tokenizers = {"qwen3vl": _FakeTokenizer()}
        model.transformer = _FakeTransformer()
        model.vae = _FakeVAE()
        model.text_encoder = _FakeTextEncoder()
        model.prompt_cache = {}
        model.tiling_config = None
        model.bits = None
        model.lora_paths = None
        model.lora_scales = None
        return model


@pytest.mark.fast
def test_generate_image_passes_denoised_to_declaring_callback():
    model = _Fixtures.krea2_model()
    callback = _RecordingCallback()
    model.callbacks.register(callback)

    model.generate_image(
        seed=1, prompt="x", num_inference_steps=2, height=64, width=64, guidance=1.0, scheduler="euler"
    )

    # The fake transformer predicts v = 1 everywhere, so the step-0 prediction is
    # exactly the seed's starting noise minus sigma_0, and it differs from the step output.
    assert [t for t, *_ in callback.calls] == [0, 1]
    t, latents_out, denoised, config = callback.calls[0]
    assert isinstance(denoised, mx.array)
    expected = Krea2LatentCreator.create_noise(1, 64, 64) - config.scheduler.sigmas[0]
    assert mx.allclose(denoised, expected).item()
    assert not mx.allclose(denoised, latents_out).item()
