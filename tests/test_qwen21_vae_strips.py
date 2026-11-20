import mlx.core as mx
import pytest

from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.model.qwen21_vae.vae import Decoder, QwenImage21VAE


@pytest.mark.fast
class TestQwen21VaeStrips:
    # A small decoder with the real layout (block count, upsamplers, residual blocks), so the context rows
    # measured on the real VAE also cover it. Random weights; the strips must match the full-image decode.
    @staticmethod
    def _vae() -> QwenImage21VAE:
        mx.random.seed(0)
        vae = QwenImage21VAE(
            {"is_residual": True, "patch_size": None, "base_dim": 8, "decoder_base_dim": 8, "z_dim": 64}
        )
        mx.eval(vae.parameters())
        return vae

    def test_strip_rows_follow_the_width_budget(self):
        assert Decoder.strip_rows(256) == 64
        assert Decoder.strip_rows(512) == 32
        assert Decoder.strip_rows(1024) == 16
        assert Decoder.strip_rows(4096) == 16

    def test_strip_decode_matches_full_image_decode(self):
        vae = self._vae()
        latents = mx.random.normal((1, 64, 1, 52, 3))
        assert vae.decode_mode == VAEBlockDecoder.LOW_MEMORY
        strips = vae.decode(latents)
        vae.decode_mode = VAEBlockDecoder.FULL_PRECISION
        full = vae.decode(latents)
        mx.eval(full, strips)
        assert strips.shape == full.shape == (1, 4, 1, 52 * 16, 3 * 16)
        assert mx.abs(strips - full).max().item() < 1e-4

    def test_short_image_decodes_in_one_piece(self):
        vae = self._vae()
        latents = mx.random.normal((1, 64, 1, 8, 4))
        strips = vae.decode(latents)
        vae.decode_mode = VAEBlockDecoder.FULL_PRECISION
        assert mx.array_equal(strips, vae.decode(latents))

    def test_nchw_vae_strips_match_full_image_decode(self):
        # Qwen21VAE (the text-to-image pipeline) runs the decoder in the NCHW layout: rows are axis 2
        mx.random.seed(0)
        vae = Qwen21VAE({"is_residual": True, "patch_size": None, "base_dim": 8, "decoder_base_dim": 8, "z_dim": 64})
        mx.eval(vae.parameters())
        latents = mx.random.normal((1, 64, 1, 52, 3))
        assert vae.decode_mode == VAEBlockDecoder.LOW_MEMORY
        strips = vae.decode(latents)
        vae.decode_mode = VAEBlockDecoder.FULL_PRECISION
        full = vae.decode(latents)
        mx.eval(full, strips)
        assert strips.shape == full.shape == (1, 3, 52 * 16, 3 * 16)
        assert mx.abs(strips - full).max().item() < 1e-4


@pytest.mark.fast
@pytest.mark.parametrize("vae_type", [QwenImage21VAE, Qwen21VAE])
def test_deeper_vae_decode_keeps_the_full_strip_context(vae_type):
    with mx.stream(mx.cpu):
        mx.random.seed(0)
        vae = vae_type(
            {"is_residual": True, "patch_size": None, "base_dim": 8, "decoder_base_dim": 8, "num_res_blocks": 4}
        )
        latents = mx.random.normal((1, 64, 1, 64, 3))
        strips = vae.decode(latents)
        vae.decode_mode = VAEBlockDecoder.FULL_PRECISION
        assert mx.array_equal(strips, vae.decode(latents))


def test_qwen_clis_describe_the_tiling_flags_and_the_decode_precision():
    from mflux.models.qwen21.cli import qwen21_controlnet_generate, qwen21_edit_generate, qwen21_generate

    # Only the text-to-image CLI tiles an encode (its --image-path input); edit and ControlNet encode untiled.
    for flag in ("--vae-tiling", "--vae-tile-size"):
        assert flag in qwen21_generate.CONDITIONAL_OPTIONS
        assert flag in qwen21_edit_generate.IGNORED_OPTIONS and flag not in qwen21_edit_generate.CONDITIONAL_OPTIONS
        assert flag in qwen21_controlnet_generate.IGNORED_OPTIONS
        assert flag not in qwen21_controlnet_generate.CONDITIONAL_OPTIONS
    # Strips and the full-image decode keep the same precision.
    for cli in (qwen21_generate, qwen21_edit_generate, qwen21_controlnet_generate):
        action = next(a for a in cli.build_parser()._actions if "--vae-full-precision" in a.option_strings)
        assert "bf16" not in action.help and "fp32" not in action.help
