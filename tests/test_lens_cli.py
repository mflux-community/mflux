import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.lens.cli.lens_generate import IGNORED_OPTIONS, build_parser


@pytest.mark.fast
class TestLensCli:
    def test_parser_builds_and_declares_ignored_options(self):
        parser = build_parser()
        assert parser is not None
        assert "--guidance" in IGNORED_OPTIONS
        assert "--negative-prompt" in IGNORED_OPTIONS

    def test_model_config_resolves_lens_aliases(self):
        for alias in ("lens-turbo", "lens"):
            config = ModelConfig.from_name(alias)
            assert config.model_name == "Comfy-Org/Lens"
            assert config.supports_guidance is False

    def test_defaults_are_turbo(self):
        config = ModelConfig.lens_turbo()
        assert config.requires_sigma_shift is True
        assert config.max_sequence_length == 512
