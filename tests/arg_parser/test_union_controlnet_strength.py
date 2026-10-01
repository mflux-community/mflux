import sys

import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.z_image.cli import z_image_turbo_generate_controlnet


def _parse(monkeypatch, argv: list[str]):
    # parse_args reads sys.argv itself; patch it for this call only so nothing leaks to other tests.
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-controlnet", "--prompt", "p", *argv])
    return z_image_turbo_generate_controlnet.build_parser().parse_args()


@pytest.mark.fast
def test_union_controlnet_global_strength_defaults_to_one(monkeypatch):
    # The effective scale is the global multiplier times the control's own strength. With the
    # FLUX ControlNet's 0.4 as the default, a 0.85 control ran at 0.34, below the range where the
    # Union checkpoint follows its hint, and the picture ignored the control (#721).
    args = _parse(monkeypatch, ["--control", "canny:a.png:0.85"])

    assert ui_defaults.UNION_CONTROLNET_STRENGTH == 1.0
    assert args.controlnet_strength == pytest.approx(1.0)
    spec = z_image_turbo_generate_controlnet.ZImageTurboControlnetCommand.parse_control_spec(args.control[0])
    assert spec.strength == pytest.approx(0.85)
    assert args.controlnet_strength * spec.strength == pytest.approx(0.85)


@pytest.mark.fast
def test_union_controlnet_global_strength_can_still_be_set(monkeypatch):
    args = _parse(monkeypatch, ["--control", "pose:a.png", "--controlnet-strength", "0.5"])

    assert args.controlnet_strength == pytest.approx(0.5)
    assert z_image_turbo_generate_controlnet.ZImageTurboControlnetCommand.parse_control_spec(
        args.control[0]
    ).strength == pytest.approx(1.0)


@pytest.mark.fast
def test_flux_controlnet_default_is_untouched():
    assert ui_defaults.CONTROLNET_STRENGTH == 0.4
