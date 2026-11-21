import warnings

from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions

# Qwen-Image-2.1 decodes in row strips by default (Decoder.STRIP_FROM). Unlike the VAEBlockDecoder models, both
# decodes keep the same precision, and only the text-to-image CLI tiles an encode (that of its --image-path input):
# the edit and ControlNet pipelines encode their input images directly.
FULL_PRECISION_HELP = (
    "Decode the VAE on the full image instead of in row strips. Both decodes use the same precision; the strips "
    "only bound the memory (at 1536x1536: 8.8 GB instead of 35.6 GB)."
)
DECODE = "The decode does not use tiles: it runs in row strips, or on the full image with --vae-full-precision."
IGNORED_TILING = {flag: f"This model encodes its input images without tiles. {DECODE}" for flag in VAEDecodeOptions.TILING_FLAGS}  # fmt: skip
ENCODE_TILING = f"--vae-tiling and --vae-tile-size change only the encode of the --image-path input. {DECODE}"
CONDITIONAL_TILING = {
    flag: {"condition": "the encode of an --image-path input", "reason": ENCODE_TILING}
    for flag in VAEDecodeOptions.TILING_FLAGS
}


def warn_encode_only_tiling() -> None:
    from mflux.cli.parser.parsers import CommandLineParser

    if CommandLineParser._option_was_provided(*VAEDecodeOptions.TILING_FLAGS):
        warnings.warn(ENCODE_TILING, stacklevel=2)
