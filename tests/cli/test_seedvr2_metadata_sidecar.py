import json
import sys

import mlx.core as mx
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.seedvr2.cli import seedvr2_upscale
from mflux.utils.image_util import ImageUtil


class TestSeedVR2MetadataSidecar:
    @staticmethod
    def _run(tmp_path, monkeypatch, extra_argv: list[str], images: int = 1):
        # Drives the real CLI over `images` inputs with the model stubbed out.
        sources = []
        for index in range(images):
            source = tmp_path / f"source{index}.png"
            ImageUtil.to_pil(mx.zeros((1, 3, 64, 64))).save(source)
            sources.append(source)

        def _image_for(image_path):
            config = Config(model_config=ModelConfig.seedvr2_3b(), num_inference_steps=1, height=64, width=64)
            return ImageUtil.to_image(
                decoded_latents=mx.zeros((1, 3, 64, 64)),
                config=config,
                seed=7,
                prompt="",
                quantization=8,
                generation_time=1.0,
                image_path=image_path,
            )

        class _StubModel:
            def __init__(self, **kwargs):
                pass

            def generate_image(self, **kwargs):
                return _image_for(kwargs["image_path"])

        monkeypatch.setattr(seedvr2_upscale, "SeedVR2", _StubModel)
        monkeypatch.setattr(seedvr2_upscale.CallbackManager, "register_callbacks", lambda **kwargs: None)
        output = tmp_path / "out_{image_name}.png"
        argv = ["mflux-upscale-seedvr2", "--image-path", *[str(source) for source in sources]]
        monkeypatch.setattr(sys, "argv", argv + ["--output", str(output)] + extra_argv)

        seedvr2_upscale.main()
        return sources

    @pytest.mark.fast
    def test_every_upscaled_image_gets_a_sidecar_naming_its_source(self, tmp_path, monkeypatch):
        # The CLI took --metadata and then called result.save(path) without forwarding it, so
        # the flag ran the whole upscale and wrote nothing. Two inputs, because forwarding it
        # on the first iteration only would pass a single-image test.
        sources = TestSeedVR2MetadataSidecar._run(tmp_path, monkeypatch, ["--metadata"], images=2)

        sidecars = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*.metadata.json"))]

        assert len(sidecars) == len(sources)
        assert [sidecar["image_path"] for sidecar in sidecars] == [str(source) for source in sources]
        assert {sidecar["seed"] for sidecar in sidecars} == {7}

    @pytest.mark.fast
    def test_no_sidecar_appears_without_the_flag(self, tmp_path, monkeypatch):
        # Pins the forwarding rather than the writing: hardcoding export_json_metadata=True
        # satisfies the test above on its own.
        TestSeedVR2MetadataSidecar._run(tmp_path, monkeypatch, [])

        assert list(tmp_path.glob("*.metadata.json")) == []
