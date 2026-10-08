import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.utils.exceptions import InvalidBaseModel, ModelConfigError


class TestConfigResolutionExactMatch:
    @pytest.mark.fast
    def test_exact_alias_match(self):
        config = ConfigResolution.resolve(model_name="schnell")

        assert config.model_name == "black-forest-labs/FLUX.1-schnell"
        assert "schnell" in config.aliases

    @pytest.mark.fast
    def test_exact_alias_match_dev(self):
        config = ConfigResolution.resolve(model_name="dev")

        assert config.model_name == "black-forest-labs/FLUX.1-dev"

    @pytest.mark.fast
    def test_exact_alias_match_fibo(self):
        config = ConfigResolution.resolve(model_name="fibo")

        assert config.model_name == "briaai/FIBO"

    @pytest.mark.fast
    def test_exact_hf_name_match(self):
        config = ConfigResolution.resolve(model_name="black-forest-labs/FLUX.1-schnell")

        assert config.model_name == "black-forest-labs/FLUX.1-schnell"


class TestConfigResolutionExplicitBase:
    @pytest.mark.fast
    def test_explicit_base_model(self):
        config = ConfigResolution.resolve(model_name="my-custom-model", base_model="schnell")

        assert config.model_name == "my-custom-model"
        assert config.base_model == "black-forest-labs/FLUX.1-schnell"
        assert config.max_sequence_length == 256  # schnell's value

    @pytest.mark.fast
    def test_explicit_base_model_dev(self):
        config = ConfigResolution.resolve(model_name="org/my-finetune", base_model="dev")

        assert config.model_name == "org/my-finetune"
        assert config.base_model == "black-forest-labs/FLUX.1-dev"
        assert config.supports_guidance is True  # dev's value

    @pytest.mark.fast
    def test_invalid_base_model_raises(self):
        with pytest.raises(InvalidBaseModel):
            ConfigResolution.resolve(model_name="whatever", base_model="invalid-base")


class TestConfigResolutionInferSubstring:
    @pytest.mark.fast
    def test_infer_from_schnell_substring(self):
        config = ConfigResolution.resolve(model_name="my-schnell-finetune")

        assert config.model_name == "my-schnell-finetune"
        assert config.base_model == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_infer_from_dev_substring(self):
        config = ConfigResolution.resolve(model_name="dev-lora-something")

        assert config.model_name == "dev-lora-something"
        assert config.base_model == "black-forest-labs/FLUX.1-dev"

    @pytest.mark.fast
    def test_infer_case_insensitive(self):
        config = ConfigResolution.resolve(model_name="MY-SCHNELL-MODEL")

        assert config.base_model == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_longer_alias_preferred(self):
        # "dev-kontext" is longer than "dev", should match dev-kontext if present
        config = ConfigResolution.resolve(model_name="my-dev-kontext-model")

        assert config.base_model == "black-forest-labs/FLUX.1-Kontext-dev"

    @pytest.mark.fast
    def test_inferred_config_preserves_text_encoder_overrides(self):
        config = ConfigResolution.resolve(model_name="/models/local-flux2-klein-9b-q4")

        assert config.base_model == "black-forest-labs/FLUX.2-klein-9B"
        assert config.transformer_overrides["num_attention_heads"] == 32
        assert config.text_encoder_overrides["hidden_size"] == 4096

    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("model_name", "base_model"),
        [
            ("Qwen/Qwen-Image-Edit-2511", None),
            ("someone/Qwen-Image-Edit-2511-mlx-q8", "Qwen/Qwen-Image-Edit-2511"),
        ],
    )
    def test_a_2511_checkpoint_gets_the_2511_entry_and_its_schedule(self, model_name, base_model):
        # 2511 has its own entry: it shares 2509's schedule and adds zero_cond_t, which a 2509 does not have.
        config = ConfigResolution.resolve(model_name=model_name)

        assert config.base_model == base_model
        assert config.sigma_max_shift == 0.9
        assert config.sigma_max_seq_len == 8192
        assert config.sigma_shift_terminal == 0.02
        assert config.transformer_overrides == {"zero_cond_t": True}

    def test_a_2509_checkpoint_stays_without_zero_cond_t(self):
        assert ConfigResolution.resolve(model_name="someone/Qwen-Image-Edit-2509-q8").transformer_overrides == {}
        assert ConfigResolution.resolve(model_name="qwen-image-edit").model_name == "Qwen/Qwen-Image-Edit-2509"


class TestConfigResolutionError:
    @pytest.mark.fast
    def test_unknown_model_without_base_raises(self):
        with pytest.raises(ModelConfigError) as exc_info:
            ConfigResolution.resolve(model_name="totally-unknown-model")

        assert "Cannot infer" in str(exc_info.value)


class TestConfigResolutionIdeogram4:
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "model_name",
        [
            "ideogram4",
            "ideogram4-fp8",
            "ideogram-4-fp8",
            "ideogram-4",
            "ideogram",
        ],
    )
    def test_exact_alias_match(self, model_name: str):
        config = ConfigResolution.resolve(model_name=model_name)

        assert config.model_name == "ideogram-ai/ideogram-4-fp8"
        assert model_name in config.aliases

    @pytest.mark.fast
    def test_exact_hf_name_match(self):
        config = ConfigResolution.resolve(model_name="ideogram-ai/ideogram-4-fp8")

        assert config.model_name == "ideogram-ai/ideogram-4-fp8"
        assert config.max_sequence_length == 2048
        assert config.supports_guidance is True
        assert config.requires_sigma_shift is False

    @pytest.mark.fast
    def test_infer_from_ideogram_substring(self):
        config = ConfigResolution.resolve(model_name="my-ideogram4-style-finetune")

        assert config.model_name == "my-ideogram4-style-finetune"
        assert config.base_model == "ideogram-ai/ideogram-4-fp8"
        assert config.max_sequence_length == 2048


class TestConfigResolutionKrea2:
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "model_name",
        [
            "krea-2",
            "krea2",
        ],
    )
    def test_exact_alias_match(self, model_name: str):
        config = ConfigResolution.resolve(model_name=model_name)

        assert config.model_name == "krea/Krea-2-Turbo"
        assert model_name in config.aliases

    @pytest.mark.fast
    def test_exact_hf_name_match(self):
        config = ConfigResolution.resolve(model_name="krea/Krea-2-Turbo")

        assert config.model_name == "krea/Krea-2-Turbo"
        assert config.max_sequence_length == 1024
        assert config.supports_guidance is True
        assert config.requires_sigma_shift is True
        assert config.sigma_max_shift == pytest.approx(1.15)

    @pytest.mark.fast
    def test_infer_from_krea2_substring(self):
        config = ConfigResolution.resolve(model_name="my-krea2-style-finetune")

        assert config.model_name == "my-krea2-style-finetune"
        assert config.base_model == "krea/Krea-2-Turbo"
        assert config.max_sequence_length == 1024


class TestConfigResolutionRules:
    @pytest.mark.fast
    def test_exact_match_takes_priority(self):
        # "schnell" is both an exact alias AND would match substring
        config = ConfigResolution.resolve(model_name="schnell")

        # Should return the exact config, not create a new one
        assert config.model_name == "black-forest-labs/FLUX.1-schnell"

    @pytest.mark.fast
    def test_explicit_base_overrides_inference(self):
        # Model name contains "schnell" but explicit base is "dev"
        config = ConfigResolution.resolve(model_name="schnell-style-dev", base_model="dev")

        assert config.base_model == "black-forest-labs/FLUX.1-dev"


class TestConfigResolutionSharedRepoId:
    # Several registry roots share a model_name with a ControlNet derivative, so a bare
    # repo id is ambiguous. The rule: a derived variant is always addressed by its own
    # key or alias, so the repo id belongs to the base entry. Before this rule, priority
    # order decided the tie and `Tongyi-MAI/Z-Image-Turbo` resolved to the ControlNet —
    # every resolver caller (generate CLIs, mflux-save dispatch) then built the wrong model.

    @staticmethod
    def _roots_by_shared_repo_id() -> dict[str, list]:
        roots: dict[str, list] = {}
        for config in AVAILABLE_MODELS.values():
            if config.base_model is None:
                roots.setdefault(config.model_name, []).append(config)
        return {name: entries for name, entries in roots.items() if len(entries) > 1}

    @pytest.mark.fast
    def test_a_shared_repo_id_resolves_to_the_base_variant(self):
        for repo_id, entries in self._roots_by_shared_repo_id().items():
            resolved = ConfigResolution.resolve(model_name=repo_id)
            assert resolved.controlnet_model is None, (
                f"{repo_id} resolved to the ControlNet variant {resolved.controlnet_model}"
            )
            assert resolved in entries

    @pytest.mark.fast
    def test_z_image_repo_id_resolves_to_plain_turbo(self):
        resolved = ConfigResolution.resolve(model_name="Tongyi-MAI/Z-Image-Turbo")

        assert resolved is AVAILABLE_MODELS["z-image-turbo"]
        assert ConfigResolution.resolve_key(model_name="Tongyi-MAI/Z-Image-Turbo") == "z-image-turbo"

    @pytest.mark.fast
    def test_every_root_key_and_alias_still_resolves_to_its_own_entry(self):
        # The tie-break must not leak: a ControlNet named by key or alias keeps resolving
        # to the ControlNet, so repo id, canonical key and alias agree per model. The key
        # is probed explicitly — today every root key is also an alias, but the resolver
        # only matches model_name and aliases, so a key that stopped being one would
        # silently stop resolving.
        for key, config in AVAILABLE_MODELS.items():
            if config.base_model is not None:
                continue
            for spelling in {key, *config.aliases}:
                owners = [c for c in AVAILABLE_MODELS.values() if c.base_model is None and spelling in c.aliases]
                if len(owners) > 1:
                    continue  # a genuinely shared alias resolves by priority; not this test's concern
                assert ConfigResolution.resolve(model_name=spelling) is config, (
                    f"{spelling!r} no longer resolves to {key}"
                )

    @pytest.mark.fast
    def test_root_aliases_are_unique(self):
        # The base-variant tie-break only ever decides bare repo ids because no two roots
        # claim the same alias. This pins that assumption: if an alias is ever shared, the
        # tie-break starts deciding alias matches too and needs a deliberate rule.
        owners: dict[str, list[str]] = {}
        for key, config in AVAILABLE_MODELS.items():
            if config.base_model is not None:
                continue
            for alias in config.aliases:
                owners.setdefault(alias, []).append(key)
        shared = {alias: keys for alias, keys in owners.items() if len(keys) > 1}
        assert shared == {}, f"aliases claimed by more than one root: {shared}"
