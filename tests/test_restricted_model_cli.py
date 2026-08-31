# CLIs hard-wired to one model must honour or reject --model, never silently ignore it.
# Regression tests for the bug where mflux-generate-krea2 --model dev still constructed
# krea/Krea-2-Turbo without a word of warning (same story on the z-image-turbo and both
# ernie CLIs). Each single-model CLI now routes --model through
# ConfigResolution.resolve_restricted: builtin registry names must be an alias of the
# CLI's own model, while paths and HuggingFace repo ids (which parse_args marks by
# setting model_path) keep the CLI's own config and load weights from the path, as they
# always have.

import sys
from types import SimpleNamespace

import pytest

from mflux.models.boogu.cli import boogu_image_generate
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.ernie_image.cli import ernie_image_generate, ernie_image_turbo_generate
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.ideogram4.cli import ideogram4_generate
from mflux.models.krea2.cli import krea2_generate
from mflux.models.lens.cli import lens_generate
from mflux.models.z_image.cli import z_image_generate, z_image_turbo_generate
from mflux.utils.exceptions import ModelConfigError

# (CLI module, registry key, a foreign model that must be rejected, the CLI's
# extra_keys, extra argv its parser requires beyond --prompt).
CLI_MODELS = [
    (krea2_generate, "krea-2", "dev", (), ()),
    (z_image_turbo_generate, "z-image-turbo", "dev", (), ()),
    (ernie_image_generate, "ernie-image", "ernie-image-turbo", (), ()),
    (ernie_image_turbo_generate, "ernie-image-turbo", "ernie-image", (), ()),
    (lens_generate, "lens-turbo", "dev", (), ()),
    (boogu_image_generate, "boogu-image-turbo", "dev", (), ()),
    (ideogram4_generate, "ideogram-4-fp8", "dev", (), ()),
    (z_image_generate, "z-image", "dev", z_image_generate.FAMILY_MODELS, ()),
    (flux2_generate, "flux2-klein-4b", "dev", flux2_generate.FAMILY_MODELS, ()),
    (
        flux2_edit_generate,
        "flux2-klein-4b",
        "qwen-image",
        flux2_edit_generate.FAMILY_MODELS,
        ("--image-paths", "ref.png"),
    ),
]


@pytest.mark.fast
class TestRestrictedModelConfig:
    # The CLI's own parser derives model_path from --model, so these tests go through
    # build_parser().parse_args() rather than constructing argument combinations no CLI
    # can produce.
    @staticmethod
    def _resolve_via_parser(
        monkeypatch, module, registry_key, extra_argv=(), extra_keys=(), base_argv=(), base_model=None
    ):
        model_argv = list(extra_argv)
        if base_model is not None:
            model_argv += ["--base-model", base_model]
        monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", *model_argv, *base_argv])
        args = module.build_parser().parse_args()
        return ConfigResolution.resolve_restricted(
            args.model, registry_key, model_path=args.model_path, extra_keys=extra_keys, base_model=args.base_model
        )

    @pytest.mark.parametrize(
        "module,registry_key,foreign,extra_keys,base_argv", CLI_MODELS, ids=lambda v: getattr(v, "__name__", v)
    )
    def test_omitted_model_returns_registry_entry(
        self, monkeypatch, module, registry_key, foreign, extra_keys, base_argv
    ):
        config = self._resolve_via_parser(monkeypatch, module, registry_key, extra_keys=extra_keys, base_argv=base_argv)
        assert config is AVAILABLE_MODELS[registry_key]

    @pytest.mark.parametrize(
        "module,registry_key,foreign,extra_keys,base_argv", CLI_MODELS, ids=lambda v: getattr(v, "__name__", v)
    )
    def test_all_aliases_accepted(self, monkeypatch, module, registry_key, foreign, extra_keys, base_argv):
        expected = AVAILABLE_MODELS[registry_key]
        for alias in expected.aliases:
            config = self._resolve_via_parser(
                monkeypatch, module, registry_key, ["--model", alias], extra_keys=extra_keys, base_argv=base_argv
            )
            assert config is expected

    @pytest.mark.parametrize(
        "module,registry_key,foreign,extra_keys,base_argv", CLI_MODELS, ids=lambda v: getattr(v, "__name__", v)
    )
    def test_foreign_model_rejected(self, monkeypatch, module, registry_key, foreign, extra_keys, base_argv):
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            self._resolve_via_parser(
                monkeypatch, module, registry_key, ["--model", foreign], extra_keys=extra_keys, base_argv=base_argv
            )

    @pytest.mark.parametrize(
        "module,registry_key,foreign,extra_keys,base_argv",
        [m for m in CLI_MODELS if m[3]],
        ids=lambda v: getattr(v, "__name__", v),
    )
    def test_family_siblings_accepted(self, monkeypatch, module, registry_key, foreign, extra_keys, base_argv):
        # Every sibling the CLI declares resolves to its own registry entry, not the
        # default's, so a klein-9b run really configures the 9B transformer.
        for key in extra_keys:
            config = self._resolve_via_parser(
                monkeypatch, module, registry_key, ["--model", key], extra_keys=extra_keys, base_argv=base_argv
            )
            assert config is AVAILABLE_MODELS[key]

    def test_flux2_family_covers_every_flux2_registry_entry(self):
        # The family tuples are registry-derived; this pins that derivation against a
        # future variant registered under a different naming scheme.
        flux2_keys = {key for key in AVAILABLE_MODELS if key.startswith("flux2-")}
        assert set(flux2_generate.FAMILY_MODELS) | {"flux2-klein-4b"} == flux2_keys
        assert set(flux2_edit_generate.FAMILY_MODELS) | {"flux2-klein-4b"} == flux2_keys

    def test_z_image_cli_rejects_the_controlnet_sibling(self, monkeypatch):
        # The base CLI runs z-image and z-image-turbo; the ControlNet entry has its own
        # command and needs a control image, so its alias must not slip in here.
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            self._resolve_via_parser(
                monkeypatch,
                z_image_generate,
                "z-image",
                ["--model", "z-image-controlnet"],
                extra_keys=z_image_generate.FAMILY_MODELS,
            )

    def test_krea2_raw_rejected_by_krea2_cli(self, monkeypatch):
        # Same architecture, but the generate CLI runs the Turbo checkpoint only; Raw is
        # the training base and must not be silently swapped for Turbo.
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            self._resolve_via_parser(monkeypatch, krea2_generate, "krea-2", ["--model", "krea-2-raw"])

    def test_z_image_controlnet_alias_rejected_despite_shared_repo_id(self, monkeypatch):
        # z-image-turbo and its ControlNet share model_name "Tongyi-MAI/Z-Image-Turbo";
        # identity comparison keeps the ControlNet alias out of the plain turbo CLI.
        assert (
            AVAILABLE_MODELS["z-image-turbo"].model_name
            == AVAILABLE_MODELS["z-image-turbo-controlnet-union-2.1"].model_name
        )
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            self._resolve_via_parser(
                monkeypatch, z_image_turbo_generate, "z-image-turbo", ["--model", "z-image-controlnet"]
            )

    def test_own_repo_id_keeps_cli_config(self, monkeypatch):
        # The repo id is not a builtin spelling, so parse_args routes it through
        # model_path; validation must not judge it (exact-match on this shared repo id
        # resolves to the ControlNet entry). Metadata reruns (-C) restore the repo id
        # from the sidecar and take this same path.
        config = self._resolve_via_parser(
            monkeypatch, z_image_turbo_generate, "z-image-turbo", ["--model", "Tongyi-MAI/Z-Image-Turbo"]
        )
        assert config is AVAILABLE_MODELS["z-image-turbo"]

    @pytest.mark.parametrize(
        "module,registry_key,path",
        [
            # Directory names whose substrings infer to a different model on main's
            # resolution rules; each loaded on main and must keep loading.
            (z_image_turbo_generate, "z-image-turbo", "~/models/zimage-q8"),
            (krea2_generate, "krea-2", "~/Developer/mflux/my-turbo-q8"),
            (ernie_image_turbo_generate, "ernie-image-turbo", "~/models/ernie-image-q4"),
        ],
        ids=lambda v: getattr(v, "__name__", v),
    )
    def test_local_checkpoint_path_keeps_cli_config(self, monkeypatch, module, registry_key, path):
        config = self._resolve_via_parser(monkeypatch, module, registry_key, ["--model", path])
        assert config is AVAILABLE_MODELS[registry_key]

    def test_saved_checkpoint_name_keeps_cli_config(self, monkeypatch):
        config = self._resolve_via_parser(monkeypatch, krea2_generate, "krea-2", ["--model", "my-krea-2-finetune"])
        assert config is AVAILABLE_MODELS["krea-2"]


# (flux2 CLI module, the parser argv its required --image-paths needs beyond --prompt).
FLUX2_BASE_ARGV = [
    (flux2_generate, ()),
    (flux2_edit_generate, ("--image-paths", "ref.png")),
]


@pytest.mark.fast
class TestFlux2BaseModelPassthrough:
    # --base-model is parsed and validated by the general argument set, but only the flux2
    # commands pass it to resolution. A custom checkpoint (repo id or local path) otherwise
    # always runs on the CLI default's geometry, which corrupts any family sibling whose
    # transformer differs — a community klein-9b checkpoint reshaped as a 4B dies at its
    # first attention layer.
    @pytest.mark.parametrize("module,base_argv", FLUX2_BASE_ARGV, ids=lambda v: getattr(v, "__name__", v))
    def test_explicit_base_selects_family_sibling(self, monkeypatch, module, base_argv):
        config = TestRestrictedModelConfig._resolve_via_parser(
            monkeypatch,
            module,
            "flux2-klein-4b",
            extra_argv=["--model", "mlx-community/flux2-klein-9b-8bit"],
            extra_keys=module.FAMILY_MODELS,
            base_model="flux2-klein-9b",
            base_argv=base_argv,
        )
        assert config is AVAILABLE_MODELS["flux2-klein-9b"]

    @pytest.mark.parametrize("module,base_argv", FLUX2_BASE_ARGV, ids=lambda v: getattr(v, "__name__", v))
    def test_omitted_model_with_base_selects_family_sibling(self, monkeypatch, module, base_argv):
        config = TestRestrictedModelConfig._resolve_via_parser(
            monkeypatch,
            module,
            "flux2-klein-4b",
            extra_keys=module.FAMILY_MODELS,
            base_model="flux2-klein-9b",
            base_argv=base_argv,
        )
        assert config is AVAILABLE_MODELS["flux2-klein-9b"]

    @pytest.mark.parametrize("module,base_argv", FLUX2_BASE_ARGV, ids=lambda v: getattr(v, "__name__", v))
    def test_foreign_base_rejected(self, monkeypatch, module, base_argv):
        # A base naming a model outside the CLI's family must not silently load foreign
        # geometry into this pipeline.
        with pytest.raises(ModelConfigError, match="only accepts the aliases"):
            TestRestrictedModelConfig._resolve_via_parser(
                monkeypatch,
                module,
                "flux2-klein-4b",
                extra_argv=["--model", "my-local-finetune"],
                extra_keys=module.FAMILY_MODELS,
                base_model="dev",
                base_argv=base_argv,
            )

    @pytest.mark.parametrize("module,base_argv", FLUX2_BASE_ARGV, ids=lambda v: getattr(v, "__name__", v))
    def test_custom_checkpoint_without_base_keeps_cli_config(self, monkeypatch, module, base_argv):
        # No flag: unchanged behavior — the checkpoint runs on the CLI default's geometry.
        config = TestRestrictedModelConfig._resolve_via_parser(
            monkeypatch,
            module,
            "flux2-klein-4b",
            extra_argv=["--model", "mlx-community/flux2-klein-9b-8bit"],
            extra_keys=module.FAMILY_MODELS,
            base_argv=base_argv,
        )
        assert config is AVAILABLE_MODELS["flux2-klein-4b"]

    # Runs main() until model construction and stops inside stubbed generate_image (SystemExit 0), so tests can
    # assert that argument-level validation let an invocation through without loading any weights.
    @staticmethod
    def _run_main_until_generation(monkeypatch, module, *argv):
        class StubModel:
            def __init__(self, *args, **kwargs):
                pass

            def generate_image(self, **kwargs):
                raise SystemExit(0)

        monkeypatch.setattr(module, "Flux2Klein" if module is flux2_generate else "Flux2KleinEdit", StubModel)
        monkeypatch.setattr(module, "CallbackManager", SimpleNamespace(register_callbacks=lambda **_: None))
        monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", *argv])
        with pytest.raises(SystemExit) as exit_info:
            module.main()
        return exit_info.value.code

    @pytest.mark.parametrize("module,base_argv", FLUX2_BASE_ARGV, ids=lambda v: getattr(v, "__name__", v))
    def test_explicit_distilled_base_rejects_guidance(self, monkeypatch, module, base_argv):
        # Once --base-model names a distilled entry the checkpoint's lineage is known, so it must get the same guidance
        # rejection as its builtin name; before this flag existed that invocation died at model load instead of here.
        argv = (
            "--model",
            "mlx-community/flux2-klein-9b-8bit",
            "--base-model",
            "flux2-klein-9b",
            "--width",
            "512",
            "--height",
            "512",
            "--guidance",
            "3.0",
            *base_argv,
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", *argv])
        with pytest.raises(SystemExit) as exit_info:
            module.main()
        assert exit_info.value.code == 2

    @pytest.mark.parametrize("module,base_argv", FLUX2_BASE_ARGV, ids=lambda v: getattr(v, "__name__", v))
    def test_explicit_base_entry_keeps_guidance(self, monkeypatch, module, base_argv):
        # The same guard must not reject a checkpoint declared as one of the guidance-capable base entries.
        exit_code = self._run_main_until_generation(
            monkeypatch,
            module,
            "--model",
            "mlx-community/flux2-klein-base-9b-8bit",
            "--base-model",
            "flux2-klein-base-9b",
            "--width",
            "512",
            "--height",
            "512",
            "--guidance",
            "3.0",
            *base_argv,
        )
        assert exit_code == 0

    @pytest.mark.parametrize("module,base_argv", FLUX2_BASE_ARGV, ids=lambda v: getattr(v, "__name__", v))
    def test_custom_checkpoint_without_base_keeps_guidance_unjudged(self, monkeypatch, module, base_argv):
        # No --base-model: the checkpoint's lineage is unknown (it may be a klein-base fine-tune), so guidance stays unjudged.
        exit_code = self._run_main_until_generation(
            monkeypatch,
            module,
            "--model",
            "mlx-community/some-finetune",
            "--width",
            "512",
            "--height",
            "512",
            "--guidance",
            "3.0",
            *base_argv,
        )
        assert exit_code == 0
