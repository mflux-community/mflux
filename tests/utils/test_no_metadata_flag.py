import sys

import PIL.Image
import pytest

from mflux.cli.parser.parsers import CommandLineParser
from mflux.utils.image_util import ImageUtil

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def _restore_toggle():
    yield
    ImageUtil.embed_metadata_enabled = True


def _save(tmp_path, name):
    out = tmp_path / name
    ImageUtil.save_image(PIL.Image.new("RGB", (8, 8), (10, 20, 30)), out, metadata={"seed": 7}, overwrite=True)
    return out


def _user_comment(path):
    # The embed lands as the PNG eXIf chunk; read it back through PIL. UserComment
    # is tag 0x9286 inside the Exif IFD (0x8769).
    exif = PIL.Image.open(path).getexif()
    return exif.get_ifd(0x8769).get(0x9286)


def test_metadata_embedded_by_default(tmp_path):
    out = _save(tmp_path, "default.png")
    comment = _user_comment(out)
    assert comment and b"seed" in comment


def test_toggle_suppresses_the_embed(tmp_path):
    ImageUtil.embed_metadata_enabled = False
    out = _save(tmp_path, "clean.png")
    assert _user_comment(out) is None


def test_flag_parses_and_flips_the_toggle(monkeypatch, tmp_path):
    parser = CommandLineParser(description="t")
    parser.add_image_generator_arguments()
    parser.add_output_arguments()
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x", "--no-metadata"])
    args = parser.parse_args()
    assert args.no_metadata is True
    assert ImageUtil.embed_metadata_enabled is False
    out = _save(tmp_path, "via_flag.png")
    assert _user_comment(out) is None


def test_without_the_flag_the_toggle_stays_on(monkeypatch):
    parser = CommandLineParser(description="t")
    parser.add_image_generator_arguments()
    parser.add_output_arguments()
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x"])
    parser.parse_args()
    assert ImageUtil.embed_metadata_enabled is True
