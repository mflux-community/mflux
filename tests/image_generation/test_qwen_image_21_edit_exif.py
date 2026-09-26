"""EXIF-orientation regression test for the Qwen-Image-2.1 edit path.

A portrait-displayed image stored landscape (EXIF orientation 6) must reach both the
vision preprocessor and the VAE rotated to what the viewer sees; RGBA alpha is
preserved for the VAE while the vision path sees a white-composited RGB copy.
All heavy models are stubbed -- no weights required.
"""

import io
import re

import mlx.core as mx
import numpy as np
import pytest
from PIL import (
    Image as PILImage,
    ImageOps,
)

from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.tokenizer.qwen21_image_processor import Qwen21ImageProcessor
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit


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
    """Splits on the special token: each <|image_pad|> becomes exactly one 151655."""

    def __init__(self):
        self.tokenizer = self

    def __call__(self, text, add_special_tokens=False, **kw):
        ids = []
        parts = text.split("<|image_pad|>")
        for i, chunk in enumerate(parts):
            ids += [1] * len(re.findall(r"\S+", chunk))
            if i < len(parts) - 1:
                ids.append(151655)
        return {"input_ids": [ids]}


class _FakeVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, tensor):
        self.inputs.append(mx.array(tensor.astype(mx.float32)))
        h, w = int(tensor.shape[2]), int(tensor.shape[3])
        return mx.zeros((1, 64, h // 16, w // 16), mx.bfloat16)

    def decode(self, latents):
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        h, w = int(latents.shape[2]) * 16, int(latents.shape[3]) * 16
        return mx.zeros((1, 3, h, w), mx.bfloat16)


class _FakeTE:
    def forward_vl(self, input_ids, pixel_values=None, image_grid_thw=None, image_token_id=151655):
        mask = input_ids == image_token_id
        return mx.zeros((1, int(input_ids.shape[1]), 4096), mx.bfloat16), mask


class _FakeTF:
    def __call_edit__(self, t, config, target_latents, layout, kv_cache=None, kv_cache_mode=None, step_cache=None):
        target = (config.height // 16) * (config.width // 16)
        return mx.zeros((1, target, 64), mx.bfloat16)


def _make_stub_model(captured):
    model = QwenImage21Edit.__new__(QwenImage21Edit)
    model.model_config = ModelConfig.qwen_image_21()
    model.callbacks = _FakeCallbacks()
    model.tiling_config = None
    model.bits = None
    model.tokenizers = {"qwen21": _FakeTokWrap()}
    model.vae = _FakeVAE()
    model.transformer = _FakeTF()
    model.text_encoder = _FakeTE()
    captured["vae_inputs"] = model.vae.inputs
    return model


def _make_orientation6_fixture(tmp_path):
    """Landscape-stored asymmetric PNG with EXIF orientation 6, plus the ground-truth
    image as it should display (orientation applied)."""
    stored = PILImage.new("L", (64, 32), 255)  # white background
    from PIL import ImageDraw as PILImageDraw

    draw = PILImageDraw.Draw(stored)
    draw.rectangle([0, 0, 20, 15], fill=0)  # black marker in the top-left corner
    stored = stored.convert("RGB")
    displayed = ImageOps.exif_transpose(stored)

    exif = PILImage.Exif()
    exif[274] = 6  # orientation 6: rotate 90 CW to display
    buf6 = io.BytesIO()
    stored.save(buf6, format="PNG", exif=exif.tobytes())
    path = tmp_path / "ref_o6.png"
    path.write_bytes(buf6.getvalue())
    # ground truth = what a viewer sees; must come from the SAVED bytes -- an
    # in-memory image carries no EXIF, so exif_transpose on `stored` is a no-op
    displayed = ImageOps.exif_transpose(PILImage.open(io.BytesIO(buf6.getvalue())))
    return str(path), stored, displayed


@pytest.mark.fast
def test_exif_orientation_reaches_both_encoders(monkeypatch, tmp_path) -> None:
    path6, stored, displayed = _make_orientation6_fixture(tmp_path)

    captured = {"preprocess": [], "vae": [], "call_edit": []}

    def fake_preprocess(self, images, resized_height=None, resized_width=None):
        captured["preprocess"].append([(im.size, im.mode) for im in images])
        grids = np.array([[1, im.size[1] // 16, im.size[0] // 16] for im in images], dtype=np.int32)
        n = sum(g[1] * g[2] for g in grids)
        return mx.zeros((n, 1536), mx.float32), mx.array(grids)

    def fake_encode(self, tensor):
        captured["vae"].append(mx.array(tensor.astype(mx.float32)))
        return mx.zeros((1, 64, int(tensor.shape[2]) // 16, int(tensor.shape[3]) // 16), mx.bfloat16)

    def fake_call_edit(self, t, config, target_latents, layout, kv_cache=None, kv_cache_mode=None):
        captured["call_edit"].append(layout)
        target = (config.height // 16) * (config.width // 16)
        return mx.zeros((1, target, 64), mx.bfloat16)

    monkeypatch.setattr(Qwen21ImageProcessor, "preprocess", fake_preprocess)
    monkeypatch.setattr(Qwen21VAE, "encode", fake_encode)
    monkeypatch.setattr(Qwen21Transformer, "__call_edit__", fake_call_edit)

    model = _make_stub_model(captured)

    model.generate_image(
        seed=1, prompt="p", image_paths=[str(path6)], num_inference_steps=1, width=64, height=64, use_kv_cache=False
    )

    # vision path: rotated to portrait, RGB (white-composited; source had no alpha)
    [(vis_entry,)] = captured["preprocess"]  # one condition image
    vis_size, vis_mode = vis_entry
    assert vis_size == (736, 1440) and vis_mode == "RGB"

    # VAE path: RGBA channels, portrait shape, rotated content. Rotation moves the
    # top-left marker to the top-right corner of the (portrait) tensor.
    vae_t = captured["vae_inputs"][-1]  # (1, 4, 1440, 736)
    assert vae_t.shape == (1, 4, 1440, 736)
    vae_img = np.array(vae_t[0].astype(mx.float32).transpose(1, 2, 0))  # (1440, 736, 4) in [-1, 1]
    lum = vae_img[..., :3].mean(axis=2)  # (1440, 736)
    h, w = lum.shape
    top_left = lum[: h // 8, : w // 8].mean()
    top_right = lum[: h // 8, -w // 8 :].mean()
    assert top_right < top_left  # marker top-right => rotated 90 CW, not 90 CCW
    disp = np.array(displayed.convert("RGBA").resize((736, 1440)), dtype=np.float32) / 127.5 - 1.0
    assert np.abs(vae_img - disp).mean() < 0.1  # content matches the displayed orientation
    assert np.allclose(vae_img[..., 3], 1.0)  # opaque alpha preserved
