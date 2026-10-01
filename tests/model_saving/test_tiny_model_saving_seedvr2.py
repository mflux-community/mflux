import pytest
from mlx import nn

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.seedvr2.cli.seedvr2_upscale import _resolve_seedvr2_model
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition3B
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class TestTinySeedVR2ModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # transformer and vae both declare hf_subdir="." (SeedVR2's real repo keeps both
        # files flat at root, told apart on load by weight_files). Saving both to one
        # directory clobbered the first's shards and index, so the transformer reloaded with
        # the VAE's weights (#621). This exercises SeedVR2's own weight definition end to end
        # through ModelSaver -> WeightLoader -> WeightApplier at tiny dimensions.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=SeedVR2WeightDefinition3B,
            make_components=TestTinySeedVR2ModelSaving._tiny_components,
            base_path=tmp_path / "seedvr2_tiny_q8",
            bits=8,
        )

    @staticmethod
    def _tiny_components():
        # vid_in_channels=16 (not the default 33) so PatchIn's proj is Linear(16*1*2*2=64, dim),
        # a multiple of 64 that SeedVR2's predicate quantizes. vid_dim=64 with one 64-wide head
        # keeps every attention Linear a multiple of 64. VAE block_out_channels=(64, 64) satisfies
        # both its GroupNorm(32) and the 64 quantization stride.
        return {
            "transformer": SeedVR2Transformer(
                vid_in_channels=16,
                vid_dim=64,
                txt_in_dim=64,
                heads=1,
                head_dim=64,
                num_layers=2,
                mm_layers=1,
            ),
            "vae": SeedVR2VAE(block_out_channels=(64, 64), latent_channels=16),
        }


class TestTinySeedVR2SavedVariant:
    @pytest.mark.fast
    @pytest.mark.parametrize(("variant", "num_layers"), [("seedvr2-3b", 32), ("seedvr2-7b", 36)])
    def test_saved_checkpoint_resolves_to_its_variant(self, tmp_path, variant, num_layers):
        # 3B and 7B share one repo id and a saved checkpoint keeps no source file names, so
        # the upscale CLI must tell them apart from the saved weights alone (#790).
        model = SeedVR2.__new__(SeedVR2)
        nn.Module.__init__(model)
        model.model_config = AVAILABLE_MODELS[variant]
        model.bits = None
        model.transformer = SeedVR2Transformer(
            vid_in_channels=16,
            vid_dim=64,
            txt_in_dim=64,
            heads=1,
            head_dim=64,
            num_layers=num_layers,
            mm_layers=1,
        )
        model.vae = SeedVR2VAE(block_out_channels=(64, 64), latent_channels=16)

        saved = tmp_path / "my-upscaler"
        model.save_model(str(saved))

        model_config, model_path = _resolve_seedvr2_model(model_arg=str(saved), model_path=str(saved))
        assert variant in model_config.aliases
        assert model_path == str(saved)
