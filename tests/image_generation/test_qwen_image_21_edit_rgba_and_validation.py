"""Weight-free tests for RGBA output, argument pre-validation, and the 10-image cap."""

import mlx.core as mx
import pytest
from PIL import Image as PILImage

from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.cli import qwen21_edit_generate as cli
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.utils.image_util import ImageUtil


class _FakeCtx:
    def before_loop(self, latents):
        pass

    def in_loop(self, t, latents, **kw):
        pass

    def after_loop(self, latents):
        pass

    def interruption(self, t, latents):
        pass


class _FakeCallbacks:
    def start(self, **kw):
        return _FakeCtx()


class _FakeTokWrap:
    def __init__(self):
        self.tokenizer = self

    def __call__(self, text, add_special_tokens=False, **kw):
        import re

        ids = []
        parts = text.split("<|image_pad|>")
        for i, chunk in enumerate(parts):
            ids += [1] * len(re.findall(r"\S+", chunk))
            if i < len(parts) - 1:
                ids.append(151655)
        return {"input_ids": [ids]}

    def decode(self, ids, **kw):
        return "ok"


class _StubVAE:
    """Decode honors keep_alpha like the real VAE: 4 channels when requested."""

    def encode(self, tensor):
        h, w = int(tensor.shape[2]), int(tensor.shape[3])
        return mx.zeros((1, 64, h // 16, w // 16), mx.bfloat16)

    def decode(self, latents, keep_alpha=False):
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        h, w = int(latents.shape[2]) * 16, int(latents.shape[3]) * 16
        channels = 4 if keep_alpha else 3
        return mx.zeros((1, channels, h, w), mx.bfloat16)


class _FakeTE:
    def forward_vl(self, input_ids, pixel_values=None, image_grid_thw=None, image_token_id=151655):
        return mx.zeros((1, int(input_ids.shape[1]), 4096), mx.bfloat16), input_ids == image_token_id


class _RecordingTF:
    transformer_blocks = [0]

    def __call_edit__(self, t, config, target_latents, layout, kv_cache=None, kv_cache_mode=None, step_cache=None):
        return mx.zeros((1, (config.height // 16) * (config.width // 16), 64), mx.bfloat16)


def _stub_model(tmp_path):
    model = QwenImage21Edit.__new__(QwenImage21Edit)
    model.model_config = ModelConfig.qwen_image_21()
    model.callbacks = _FakeCallbacks()
    model.tiling_config = None
    model.bits = None
    model.tokenizers = {"qwen21": _FakeTokWrap()}
    model.vae = _StubVAE()
    model.transformer = _RecordingTF()
    model.text_encoder = _FakeTE()
    return model


@pytest.mark.fast
def test_rgba_output_keeps_alpha_and_pil_mode(tmp_path) -> None:
    ref = tmp_path / "ref.png"
    PILImage.new("RGB", (128, 128), (10, 200, 30)).save(ref)
    model = _stub_model(tmp_path)
    captured = {}

    def fake_to_image(**kwargs):
        captured.update(kwargs)
        img = ImageUtil.to_pil(kwargs["decoded_latents"])
        captured["pil_mode"] = img.mode
        return img

    import mflux.models.qwen21.variants.edit.qwen_image_21_edit as edit_module

    orig = ImageUtil.to_image
    edit_module.ImageUtil.to_image = staticmethod(fake_to_image)
    try:
        model.generate_image(
            seed=1,
            prompt="p",
            image_paths=[str(ref)],
            num_inference_steps=1,
            width=128,
            height=128,
            rgba_output=True,
            use_kv_cache=False,
        )
        assert captured["decoded_latents"].shape[1] == 4  # RGBA channels kept
        assert captured["pil_mode"] == "RGBA"

        model.generate_image(
            seed=1, prompt="p", image_paths=[str(ref)], num_inference_steps=1, width=128, height=128, use_kv_cache=False
        )
        assert captured["decoded_latents"].shape[1] == 3  # default still drops alpha
    finally:
        edit_module.ImageUtil.to_image = orig


@pytest.mark.fast
def test_more_than_ten_condition_images_rejected(tmp_path) -> None:
    ref = tmp_path / "ref.png"
    PILImage.new("RGB", (64, 64)).save(ref)
    model = _stub_model(tmp_path)
    with pytest.raises(ValueError, match="10 condition images"):
        model.generate_image(
            seed=1,
            prompt="p",
            image_paths=[str(ref)] * 11,
            num_inference_steps=1,
            width=64,
            height=64,
            use_kv_cache=False,
        )


# ------------------------------------------------------- CLI pre-load validation


def _run_cli(monkeypatch, tmp_path, extra):
    import sys

    src = tmp_path / "ref.png"
    PILImage.new("RGB", (128, 128)).save(src)
    out = tmp_path / "out_{seed}.png"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mflux-generate-qwen-2.1-edit", "--image-paths", str(src), "--prompt", "p", "--output", str(out), *extra],
    )

    parser = cli.build_parser()
    args = parser.parse_args()
    return parser, args


@pytest.mark.fast
def test_cli_validation_rejects_bad_args_before_load(monkeypatch, tmp_path, capsys) -> None:
    import sys as _sys

    src = tmp_path / "ref.png"
    PILImage.new("RGB", (128, 128)).save(src)

    def run_cli(extra):
        monkeypatch.setattr(
            _sys,
            "argv",
            [
                "mflux-generate-qwen-2.1-edit",
                "--image-paths",
                str(src),
                "--prompt",
                "p",
                "--output",
                str(tmp_path / "out_{seed}.png"),
                *extra,
            ],
        )
        parser = cli.build_parser()
        args = parser.parse_args()
        return parser, args

    # strength out of range
    parser, args = run_cli(["--strength", "1.5"])
    with pytest.raises(SystemExit):
        cli.validate_args(parser, args)
    assert "strength" in capsys.readouterr().err

    # non-default scheduler
    parser, args = run_cli(["--scheduler", "deis"])
    with pytest.raises(SystemExit):
        cli.validate_args(parser, args)
    assert "linear Euler" in capsys.readouterr().err

    # guidance below 1
    parser, args = run_cli(["--guidance", "0.5"])
    with pytest.raises(SystemExit):
        cli.validate_args(parser, args)
    assert "guidance" in capsys.readouterr().err

    # rgba output to a JPEG path
    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "mflux-generate-qwen-2.1-edit",
            "--image-paths",
            str(src),
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "o.jpg"),
            "--rgba-output",
        ],
    )
    parser = cli.build_parser()
    args = parser.parse_args()
    with pytest.raises(SystemExit):
        cli.validate_args(parser, args)
    assert "PNG" in capsys.readouterr().err

    # missing condition image
    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "mflux-generate-qwen-2.1-edit",
            "--image-paths",
            str(tmp_path / "nope.png"),
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "o.png"),
        ],
    )
    parser = cli.build_parser()
    args = parser.parse_args()
    with pytest.raises(SystemExit):
        cli.validate_args(parser, args)
    assert "not found" in capsys.readouterr().err


@pytest.mark.fast
def test_cli_validation_accepts_valid_args(monkeypatch, tmp_path) -> None:
    parser, args = _run_cli(monkeypatch, tmp_path, [])
    cli.validate_args(parser, args)  # no SystemExit
