import json
import sys
import warnings
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mflux.cli.capabilities import describe_command
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.ernie_image import ernie_image_initializer
from mflux.models.ernie_image.cli import ernie_image_generate, ernie_image_turbo_generate
from mflux.models.fibo2 import fibo2_initializer
from mflux.models.fibo2.cli import fibo2_edit_generate, fibo2_generate
from mflux.models.flux2 import flux2_initializer
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.flux2.model.flux2_vae import small_decoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux3.cli import flux3_edit_generate, flux3_generate
from mflux.models.ideogram4 import ideogram4_initializer
from mflux.models.ideogram4.cli import ideogram4_generate
from mflux.models.lens.cli import lens_generate
from mflux.models.z_image import z_image_initializer
from mflux.models.z_image.cli import z_image_generate, z_image_turbo_generate, z_image_turbo_generate_controlnet
from mflux.models.z_image.model.z_image_vae.vae import VAE as ZImageVAE

# 16x16 latent -> 128x128 image; 256 attention tokens.
ZIMAGE_LATENT_SHAPE = (1, 16, 16, 16)


@pytest.fixture(scope="module")
def zimage_vae() -> ZImageVAE:
    mx.random.seed(0)
    vae = ZImageVAE()
    mx.eval(vae.decoder.parameters())
    return vae


def _latent() -> mx.array:
    mx.random.seed(1)
    return mx.random.normal(ZIMAGE_LATENT_SHAPE).astype(mx.bfloat16)


def _decode(vae, mode: str, latent: mx.array, tiling_config: TilingConfig | None = None) -> np.ndarray:
    vae.decode_mode = mode
    try:
        return np.array(VAEUtil.decode(vae=vae, latent=latent, tiling_config=tiling_config).astype(mx.float32))
    finally:
        vae.decode_mode = VAEBlockDecoder.LEGACY


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.fast
def test_zimage_full_precision_matches_legacy_within_rounding(zimage_vae):
    legacy = _decode(zimage_vae, VAEBlockDecoder.LEGACY, _latent())
    full_precision = _decode(zimage_vae, VAEBlockDecoder.FULL_PRECISION, _latent())

    assert full_precision.shape == legacy.shape == (1, 3, 128, 128)
    assert _relative_error(full_precision, legacy) < 1e-2


@pytest.mark.fast
def test_zimage_low_memory_strips_match_one_strip_without_seams(monkeypatch, zimage_vae):
    monkeypatch.setattr(VAEBlockDecoder, "STRIP_ROWS", 10_000)
    one_strip = _decode(zimage_vae, VAEBlockDecoder.LOW_MEMORY, _latent())
    monkeypatch.setattr(VAEBlockDecoder, "STRIP_ROWS", 16)
    monkeypatch.setattr(VAEBlockDecoder, "ATTENTION_CHUNK", 64)
    strips = _decode(zimage_vae, VAEBlockDecoder.LOW_MEMORY, _latent())

    assert _relative_error(strips, one_strip) < 2e-2
    error_by_row = np.abs(strips - one_strip).mean(axis=(0, 1, 3))
    edge_rows = [row for row in range(128) if row % 16 in (0, 15)]
    inner_rows = [row for row in range(128) if row % 16 not in (0, 15)]
    assert error_by_row[edge_rows].mean() < 1.3 * error_by_row[inner_rows].mean()


@pytest.mark.fast
def test_zimage_low_memory_stays_within_bf16_rounding_of_legacy(monkeypatch, zimage_vae):
    monkeypatch.setattr(VAEBlockDecoder, "STRIP_ROWS", 16)

    assert _relative_error(_decode(zimage_vae, VAEBlockDecoder.LOW_MEMORY, _latent()), _decode(zimage_vae, VAEBlockDecoder.LEGACY, _latent())) < 5e-2  # fmt: skip


@pytest.mark.fast
def test_vae_util_tiles_only_the_legacy_decode(monkeypatch, zimage_vae):
    calls = []
    monkeypatch.setattr("mflux.models.common.vae.vae_util.VAETiler.decode_image_tiled", lambda **kwargs: calls.append(1) or mx.zeros((1, 3, 1, 128, 128)))  # fmt: skip
    tiling = TilingConfig(vae_decode_tile_size=128)

    _decode(zimage_vae, VAEBlockDecoder.LOW_MEMORY, _latent(), tiling)
    _decode(zimage_vae, VAEBlockDecoder.FULL_PRECISION, _latent(), tiling)
    assert calls == []
    _decode(zimage_vae, VAEBlockDecoder.LEGACY, _latent(), tiling)
    assert calls == [1]


@pytest.mark.fast
def test_vae_util_keeps_legacy_for_a_vae_without_a_decode_mode():
    class PlainVAE:
        def decode(self, latent):
            return latent + 1

    assert VAEUtil.decode(vae=PlainVAE(), latent=mx.zeros((1, 3, 4, 4))).sum().item() == 48


@pytest.mark.fast
@pytest.mark.parametrize(
    ("module", "patched", "call"),
    [
        (
            flux2_initializer,
            ["Flux2Transformer", "Qwen3TextEncoder"],
            lambda init, model: init.Flux2Initializer._init_models(model),
        ),  # fmt: skip
        (
            ernie_image_initializer,
            ["ErnieTransformer", "ErnieMistralTextEncoder"],
            lambda init, model: init.ErnieImageInitializer._init_models(
                model, SimpleNamespace(transformer_overrides={})
            ),
        ),  # fmt: skip
        (ideogram4_initializer, ["Ideogram4Transformer", "Qwen3TextEncoder"], None),
        (
            z_image_initializer,
            ["ZImageTransformer", "TextEncoder"],
            lambda init, model: init.ZImageInitializer._init_models(model),
        ),  # fmt: skip
        (
            fibo2_initializer,
            ["Fibo2Transformer", "Fibo2TextEncoder"],
            lambda init, model: init.Fibo2Initializer._init_models(model),
        ),  # fmt: skip
    ],
    ids=["flux2", "ernie_image", "ideogram4", "z_image", "fibo2"],
)
def test_each_model_uses_the_low_memory_decode(monkeypatch, tmp_path, module, patched, call):
    for name in patched:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    model = SimpleNamespace(model_config=SimpleNamespace(transformer_overrides={}, text_encoder_overrides={}), small_text_encoder=False)  # fmt: skip
    if module is ideogram4_initializer:
        monkeypatch.setattr(module.Ideogram4Initializer, "_transformer_config", staticmethod(lambda path: None))
        monkeypatch.setattr(module.Ideogram4Initializer, "_text_encoder_kwargs", staticmethod(lambda path: {}))
        module.Ideogram4Initializer._init_models(model, tmp_path)
    else:
        call(module, model)

    assert model.vae.decode_mode == VAEBlockDecoder.LOW_MEMORY


@pytest.mark.fast
def test_the_small_decoder_keeps_the_decode_mode(monkeypatch):
    vae = Flux2VAE(decoder_block_out_channels=(32, 32, 64, 64))
    vae.decode_mode = VAEBlockDecoder.LOW_MEMORY
    monkeypatch.setattr(small_decoder.WeightLoader, "load_single", lambda *args, **kwargs: SimpleNamespace(components={}, meta_data=None))  # fmt: skip
    monkeypatch.setattr(small_decoder.Flux2SmallDecoderLoader, "check_weights", staticmethod(lambda *args: None))
    monkeypatch.setattr(small_decoder.WeightApplier, "apply_and_quantize_single", lambda *args, **kwargs: None)
    monkeypatch.setattr(small_decoder.mx, "eval", lambda *args: None)

    assert small_decoder.Flux2SmallDecoderLoader.with_small_decoder(vae).decode_mode == VAEBlockDecoder.LOW_MEMORY


@pytest.mark.fast
@pytest.mark.parametrize(
    ("mode", "expected"),
    [(VAEBlockDecoder.FULL_PRECISION, {"vae_full_precision": True}), (VAEBlockDecoder.LOW_MEMORY, {}), (None, {})],
)
def test_metadata_records_the_full_precision_decode_only_when_on(mode, expected):
    model = SimpleNamespace() if mode is None else SimpleNamespace(vae=SimpleNamespace(decode_mode=mode))

    assert VAEDecodeOptions.generation_parameters(model) == expected


# (module, status of --vae-tiling, argv the CLI needs besides --prompt)
CLIS = [
    (flux2_generate, "conditional", []),
    (flux2_edit_generate, "conditional", ["--image-paths", "{ref}"]),
    (ernie_image_generate, "conditional", []),
    (ernie_image_turbo_generate, "conditional", []),
    (lens_generate, "ignored", []),
    (ideogram4_generate, "ignored", []),
    (z_image_generate, "conditional", []),
    (z_image_turbo_generate, "conditional", []),
    (z_image_turbo_generate_controlnet, "conditional", ["--control", "canny:{ref}"]),
    (flux3_generate, "ignored", []),
    (flux3_edit_generate, "conditional", ["--image-paths", "{ref}"]),
    (fibo2_generate, "ignored", []),
    (fibo2_edit_generate, "conditional", ["--image-paths", "{ref}"]),
]


def _cli_id(case) -> str:
    return case[0].__name__.rsplit(".", 1)[-1]


@pytest.mark.fast
@pytest.mark.parametrize(("cli", "status", "extra"), CLIS, ids=[_cli_id(c) for c in CLIS])
def test_capabilities_report_the_vae_tiling_status(cli, status, extra):
    entry = describe_command("cmd", cli.__name__)
    options = {option["flag"]: option["status"] for option in entry["options"]}

    assert options["--vae-tiling"] == status
    assert options["--vae-tile-size"] == status
    assert options["--vae-full-precision"] == "honored"


@pytest.mark.fast
@pytest.mark.parametrize(("cli", "status", "extra"), CLIS, ids=[_cli_id(c) for c in CLIS])
@pytest.mark.parametrize(("recorded", "argv_flag", "expected"), [(True, [], True), (True, ["--no-vae-full-precision"], False), (None, ["--vae-full-precision"], True), (None, [], False)])  # fmt: skip
def test_the_parser_reads_the_flag_and_replays_it(monkeypatch, tmp_path, cli, status, extra, recorded, argv_flag, expected):  # fmt: skip
    reference = tmp_path / "ref.png"
    reference.touch()
    sidecar = tmp_path / "image.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a fox", "seed": 1, "steps": 4} | ({} if recorded is None else {"vae_full_precision": recorded})))  # fmt: skip
    replay = ["--config-from-metadata", str(sidecar)]
    if cli is z_image_turbo_generate_controlnet:
        # This CLI has no --config-from-metadata. Only the flag itself applies.
        if recorded is not None:
            pytest.skip("no --config-from-metadata in this CLI")
        replay = []
    argv = [
        "cmd",
        *replay,
        "--prompt",
        "a fox",
        *[a.format(ref=reference) for a in extra],
        *argv_flag,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert cli.build_parser().parse_args().vae_full_precision is expected


@pytest.mark.fast
@pytest.mark.parametrize(("cli", "status", "extra"), CLIS, ids=[_cli_id(c) for c in CLIS])
def test_the_vae_tiling_flag_warns_that_the_decode_does_not_tile(monkeypatch, tmp_path, cli, status, extra):
    reference = tmp_path / "ref.png"
    reference.touch()
    monkeypatch.setattr(sys, "argv", ["cmd", "--prompt", "x", "--vae-tiling", *[a.format(ref=reference) for a in extra]])  # fmt: skip

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    for name in dir(cli):
        if name.endswith("Command") and hasattr(getattr(cli, name), "load"):
            monkeypatch.setattr(getattr(cli, name), "load", stop)
    for name in ("Flux2Klein", "Flux2KleinEdit", "Flux3Klein", "Flux3KleinEdit"):
        if hasattr(cli, name):
            monkeypatch.setattr(cli, name, stop)
    with warnings.catch_warnings(record=True) as caught, pytest.raises((Stop, SystemExit)):
        warnings.simplefilter("always")
        cli.main()
    assert any("row strips" in str(w.message) for w in caught)


@pytest.mark.fast
def test_apply_selects_full_precision_only_when_asked():
    model = SimpleNamespace(vae=SimpleNamespace(decode_mode=VAEBlockDecoder.LOW_MEMORY))
    VAEDecodeOptions.apply(SimpleNamespace(vae_full_precision=False), model)
    assert model.vae.decode_mode == VAEBlockDecoder.LOW_MEMORY
    VAEDecodeOptions.apply(SimpleNamespace(vae_full_precision=True), model)
    assert model.vae.decode_mode == VAEBlockDecoder.FULL_PRECISION
    VAEDecodeOptions.apply(SimpleNamespace(), SimpleNamespace())
