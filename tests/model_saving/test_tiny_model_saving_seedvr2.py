import pytest

from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
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
