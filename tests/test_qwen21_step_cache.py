import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.cli.capabilities import describe_command
from mflux.models.common.config import ModelConfig
from mflux.models.common.step_cache.step_cache import StepCache
from mflux.models.qwen21.cli import qwen21_generate
from mflux.models.qwen21.model.qwen21_transformer.qwen21_time_text_embed import Qwen21TimeTextEmbed
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21


class _StubTransformer:
    """Counts denoise calls; returns a constant prediction so the loop stays cheap."""

    def __init__(self):
        self.time_text_embed = Qwen21TimeTextEmbed(embedding_dim=64)
        self.called_steps: list[int] = []
        self.cleared = False

    def __call__(self, t, config, hidden_states, encoder_hidden_states, encoder_hidden_states_mask):
        self.called_steps.append(t)
        return mx.zeros_like(hidden_states)

    def clear_text_cache(self):
        self.cleared = True


def _stub_model() -> QwenImage21:
    # Bypass weight loading: generate_image only needs these attributes.
    model = QwenImage21.__new__(QwenImage21)
    embeds = mx.zeros((1, 4, 64), dtype=ModelConfig.precision)
    model.__dict__.update(
        transformer=_StubTransformer(),
        model_config=ModelConfig.qwen_image_21(),
        prompt_cache={"a cat": (embeds, mx.ones((1, 4)))},
        tokenizers={"qwen21": None},
        text_encoder=None,
        callbacks=CallbackRegistry(),
        tiling_config=None,
        vae=None,
        bits=None,
        lora_paths=None,
        lora_scales=None,
    )
    return model


@pytest.fixture
def no_decode(monkeypatch):
    from mflux.models.qwen21.variants.txt2img import qwen_image_21 as module

    monkeypatch.setattr(module.VAEUtil, "decode", staticmethod(lambda vae, latent, tiling_config: latent))
    monkeypatch.setattr(
        module.Qwen21LatentCreator, "unpack_latents", staticmethod(lambda latents, height, width: latents)
    )
    captured = {}

    def fake_to_image(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(module.ImageUtil, "to_image", staticmethod(fake_to_image))
    return captured


@pytest.mark.fast
class TestQwen21StepCacheWiring:
    def test_skipped_steps_do_not_call_the_transformer(self, no_decode):
        model = _stub_model()
        model.generate_image(
            seed=1, prompt="a cat", num_inference_steps=40, height=128, width=128, step_cache_ratio=0.25
        )
        assert len(model.transformer.called_steps) == 30  # 40 steps, 10 reused
        assert model.transformer.cleared
        assert no_decode["generation_parameters"] == {"step_cache_ratio": 0.25}

    def test_default_run_calls_the_transformer_every_step(self, no_decode):
        model = _stub_model()
        model.generate_image(seed=1, prompt="a cat", num_inference_steps=40, height=128, width=128)
        assert model.transformer.called_steps == list(range(40))
        assert no_decode["generation_parameters"] == {}

    def test_skip_set_uses_the_timestep_embedding_signal(self, no_decode):
        model = _stub_model()
        model.generate_image(
            seed=1, prompt="a cat", num_inference_steps=40, height=128, width=128, step_cache_ratio=0.25
        )
        skipped = set(range(40)) - set(model.transformer.called_steps)
        from mflux.models.common.config.config import Config

        config = Config(
            width=128,
            height=128,
            guidance=1.0,
            scheduler="linear",
            model_config=model.model_config,
            num_inference_steps=40,
        )
        expected = StepCache.for_run(config, ratio=0.25, signal_fn=model.transformer.time_text_embed).skip_steps
        assert skipped == set(expected)

    def test_deprecated_teacache_ratio_alias_still_works(self, no_decode):
        model = _stub_model()
        model.generate_image(seed=1, prompt="a cat", num_inference_steps=40, height=128, width=128, teacache_ratio=0.25)
        assert len(model.transformer.called_steps) == 30
        assert no_decode["generation_parameters"] == {"step_cache_ratio": 0.25}

    def test_conflicting_ratio_and_alias_raise(self, no_decode):
        with pytest.raises(ValueError):
            _stub_model().generate_image(
                seed=1,
                prompt="a cat",
                num_inference_steps=40,
                height=128,
                width=128,
                step_cache_ratio=0.2,
                teacache_ratio=0.3,
            )


@pytest.mark.fast
class TestQwen21StepCacheCli:
    @pytest.mark.parametrize("flag", ["--step-cache-ratio", "--teacache-ratio"])
    def test_flag_and_alias_parse(self, monkeypatch, flag):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen-2.1", "--prompt", "x", flag, "0.25"])
        assert qwen21_generate.build_parser().parse_args().step_cache_ratio == 0.25

    @pytest.mark.parametrize("value", ["0", "1", "-0.5", "abc"])
    def test_invalid_ratio_rejected_before_loading(self, monkeypatch, value):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen-2.1", "--prompt", "x", "--step-cache-ratio", value])
        with pytest.raises(SystemExit) as exc:
            qwen21_generate.build_parser().parse_args()
        assert exc.value.code == 2

    def test_capabilities_report_the_flag_as_honored(self):
        command = describe_command("mflux-generate-qwen-2.1", qwen21_generate.__name__)
        options = {option["flag"]: option for option in command["options"]}
        assert options["--step-cache-ratio"]["status"] == "honored"
        assert options["--step-cache-ratio"]["aliases"] == ["--teacache-ratio"]

    # --config-from-metadata and -C are aliases of --config-from-conf (#703).
    @pytest.mark.parametrize("flag", ["--config-from-conf", "--config-from-metadata", "-C"])
    def test_config_from_metadata_replays_the_ratio(self, monkeypatch, tmp_path, flag):
        sidecar = tmp_path / "image.json"
        sidecar.write_text('{"prompt": "x", "seed": 3, "steps": 40, "step_cache_ratio": 0.25}')
        monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen-2.1", flag, str(sidecar)])
        assert qwen21_generate.build_parser().parse_args().step_cache_ratio == 0.25

    @pytest.mark.parametrize("bad", ["5", "0", "-0.2", '"abc"'])
    def test_invalid_metadata_ratio_rejected_during_parsing(self, monkeypatch, tmp_path, bad):
        sidecar = tmp_path / "image.json"
        sidecar.write_text(f'{{"prompt": "x", "seed": 3, "steps": 40, "step_cache_ratio": {bad}}}')
        monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen-2.1", "--config-from-metadata", str(sidecar)])
        with pytest.raises(SystemExit) as exc:
            qwen21_generate.build_parser().parse_args()
        assert exc.value.code == 2

    def test_command_line_ratio_overrides_metadata(self, monkeypatch, tmp_path):
        sidecar = tmp_path / "image.json"
        sidecar.write_text('{"prompt": "x", "seed": 3, "steps": 40, "step_cache_ratio": 0.25}')
        monkeypatch.setattr(
            sys, "argv", ["mflux-generate-qwen-2.1", "--config-from-metadata", str(sidecar), "--teacache-ratio", "0.4"]
        )
        assert qwen21_generate.build_parser().parse_args().step_cache_ratio == 0.4
