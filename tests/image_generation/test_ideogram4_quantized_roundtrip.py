"""Golden tests for Ideogram 4 checkpoints saved by `mflux-save -q`.

These prove the round trip end to end: quantize an fp8 checkpoint, save it, load it
back, and generate the image the fp8 model generates. They are skipped unless the saved
checkpoints are on disk, since the artefacts are 14-26 GB and cannot live in CI.

Produce them with:

    mflux-save --model ideogram-4-fp8 -q 8 --path /path/to/ideogram-4-mflux-q8
    mflux-save --model ideogram-4-fp8 -q 4 --path /path/to/ideogram-4-mflux-q4

then point the tests at them:

    export MFLUX_IDEOGRAM4_Q8_PATH=/path/to/ideogram-4-mflux-q8
    export MFLUX_IDEOGRAM4_Q4_PATH=/path/to/ideogram-4-mflux-q4
    pytest tests/image_generation/test_ideogram4_quantized_roundtrip.py -m slow -s

`-s` is worth passing: every comparison prints its measured mismatch, which is the
number to quote when reporting what quantization costs.

Set `MFLUX_IDEOGRAM4_GOLDEN_OUTPUT_DIR` to keep the generated images, laid out as
`<dir>/{fp8,q8,q4}/{puffin,jazz_fest}.png` — the side-by-side evidence for a PR or a
model card. Without it they go to pytest's temp directory and are eventually reaped.

Each checkpoint is loaded exactly once and generates every case, so the run costs three
model loads rather than one per comparison.

Quantized output is measured against an fp8 image generated *in the same run on the same
machine*, not against a checked-in PNG. That is deliberate: a stored reference also
measures accumulated drift in the fp8 path and across platforms, and `ImageCompare`
works at `rtol=0.1, atol=0`, which makes a one-level difference on a dark subpixel count
the same as a catastrophic one. Against a live baseline the only variable left is the
thing under test.

Note that the mismatch ratio says *different*, not *worse*. Diffusion is chaotic, so a
small weight perturbation can move a discrete decision — where a glyph lands, how a
shape resolves — and produce an image that is equally good and numerically far away.
Both quantized checkpoints measure a long way from fp8 by this metric and are
nonetheless indistinguishable from it side by side. Judging quality means looking at
the images; these numbers only detect that something moved.
"""

import gc
import json
import os
from pathlib import Path
from typing import Any

import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.ideogram4.variants import Ideogram4
from mflux.utils.image_compare import ImageCompare
from tests.image_generation.helpers.image_generation_ideogram4_test_helper import (
    ImageGeneratorIdeogram4TestHelper,
)

CHECKPOINTS = {
    8: os.environ.get("MFLUX_IDEOGRAM4_Q8_PATH"),
    4: os.environ.get("MFLUX_IDEOGRAM4_Q4_PATH"),
}

# Set this to keep the generated images somewhere findable — they are the side-by-side
# evidence for a PR or a model card. Without it they land in pytest's temp directory and
# are eventually reaped.
OUTPUT_DIR = os.environ.get("MFLUX_IDEOGRAM4_GOLDEN_OUTPUT_DIR")

# Both cases use JSON captions, and that is not stylistic. Ideogram 4's safety filter
# rejects plain prompts often enough that the README's own "A puffin standing on a
# cliff" comes back blocked, and a blocked generation is a flat grey field, not an
# error. Two blocked outputs compare almost identically, so a plain-prompt case here
# would look like a passing test that had proven nothing at all.
#
# "puffin" is the photographic control; "jazz_fest" is the existing golden caption, a
# near-black typographic poster, which is what Ideogram 4 is actually for.
PUFFIN_JSON_CAPTION: dict[str, Any] = {
    "high_level_description": "A wildlife photograph of an Atlantic puffin standing on a coastal cliff edge.",
    "style_description": {
        "aesthetics": "naturalistic, crisp, shallow depth of field",
        "lighting": "soft overcast daylight from the left",
        "medium": "photography",
        "art_style": "documentary wildlife photography, telephoto compression",
        "color_palette": ["#1B1B1B", "#F2F2F0", "#E8552D", "#8A9BA8"],
    },
    "compositional_deconstruction": {
        "background": "Soft-focus grey sea and overcast sky behind the cliff edge.",
        "elements": [
            {
                "type": "obj",
                "bbox": [300, 380, 820, 640],
                "desc": "An Atlantic puffin standing upright in profile on grass-topped rock, orange bill and feet, black back and white breast, sharply focused.",
            },
            {
                "type": "obj",
                "bbox": [780, 0, 1000, 1000],
                "desc": "Grass and lichen-covered cliff rock spanning the lower edge of the frame.",
            },
        ],
    },
}

CASES: dict[str, dict[str, Any]] = {
    "puffin": {
        "prompt": PUFFIN_JSON_CAPTION,
        "seed": 42,
        "width": 1024,
        "height": 1024,
        "preset": "V4_TURBO_12",
    },
    "jazz_fest": {
        "prompt": ImageGeneratorIdeogram4TestHelper.JAZZ_FEST_JSON_CAPTION,
        "seed": 202,
        "width": 768,
        "height": 576,
        "preset": "V4_TURBO_12",
    },
}

# Ceilings, not expectations, and per case because the two behave very differently.
# Measured on an M5 Pro, mismatch at rtol=0.1/atol=0 against a live fp8 baseline:
#
#              q8      q4
#   puffin      5.9%   38.1%
#   jazz_fest  29.2%   65.6%
#
# Read those with care. Inspected side by side, q8 and q4 are both indistinguishable
# from fp8 to the eye — so a two-thirds "mismatch" is not two-thirds wrong. Diffusion is
# chaotic: perturbing the weights slightly moves a discrete decision (where a glyph
# lands, how an edge resolves) and produces an image that is equally good and numerically
# far away. The metric counts a subpixel as mismatched when it moves 10% in relative
# terms, which on this kind of comparison tracks displacement far more than damage.
#
# The typographic case diverges roughly five times as much as the photographic one at
# q8, which is what you would expect from a model whose whole job is typography. It is
# not a dark-image artefact: the divergence is no heavier among dark subpixels than
# light ones, and the 99th-percentile difference is 80 levels at q8 and 206 at q4.
#
# These ceilings therefore catch gross regressions only. They do not certify quality,
# and no threshold on this metric could.
MISMATCH_CEILING = {
    ("puffin", 8): 0.40,
    ("puffin", 4): 0.75,
    ("jazz_fest", 8): 0.40,
    ("jazz_fest", 4): 0.75,
}

# Two fp8 runs at the same seed should land in the same place, making the figures above
# attributable to quantization alone. Kept slightly above zero rather than at it so the
# test reports a real floor rather than tripping on a single stray subpixel.
NOISE_FLOOR_CEILING = 0.005


def _generate_all(model_path: str | None, destination: Path, *, repeats: int = 1) -> dict[str, list[Path]]:
    """Load one checkpoint and generate every case from it, then release it.

    `repeats` generates each case more than once from the *same* loaded model, which is
    how the fp8 noise floor is measured without paying for a second load.
    """
    model = None
    outputs: dict[str, list[Path]] = {}
    try:
        model = Ideogram4(model_path=model_path, model_config=ModelConfig.ideogram4_fp8())
        for name, case in CASES.items():
            paths = []
            for attempt in range(repeats):
                result = model.generate_image(**case)
                path = destination / (f"{name}.png" if attempt == 0 else f"{name}_repeat{attempt}.png")
                result.save(path=path, overwrite=True)
                paths.append(path)
            outputs[name] = paths
    finally:
        model = None
        gc.collect()
        try:
            import mlx.core as mx  # noqa: PLC0415

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass
    return outputs


def _destination(tmp_path_factory, label: str) -> Path:
    if OUTPUT_DIR:
        path = Path(OUTPUT_DIR).expanduser() / label
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp(f"ideogram4_{label}")


@pytest.fixture(scope="module")
def fp8_baselines(tmp_path_factory) -> dict[str, list[Path]]:
    # Twice per case: the first is the baseline every quantized run is measured against,
    # the second establishes the noise floor that makes those measurements mean anything.
    return _generate_all(None, _destination(tmp_path_factory, "fp8"), repeats=2)


@pytest.fixture(scope="module")
def quantized_outputs(tmp_path_factory) -> dict[int, dict[str, list[Path]]]:
    outputs: dict[int, dict[str, list[Path]]] = {}
    for bits, path in CHECKPOINTS.items():
        if not path or not Path(path).exists():
            continue
        outputs[bits] = _generate_all(path, _destination(tmp_path_factory, f"q{bits}"))
    return outputs


def _edge_energy(path: Path) -> float:
    """Mean gradient magnitude — a cheap proxy for 'does this image contain anything'."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    data = np.array(Image.open(path)).astype(np.float64)
    grey = data.mean(axis=-1) if data.ndim == 3 else data
    gy, gx = np.gradient(grey)
    return float(np.mean(np.hypot(gy, gx)))


# A blocked generation comes back as a flat field rather than raising. Measured on an
# M5 Pro: a safety-filtered output scores ~0.4, a real generation ~6.9. Anything under
# this floor is not an image worth comparing.
MIN_EDGE_ENERGY = 1.5


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_fp8_baseline_was_actually_generated(case: str, fp8_baselines):
    """Guard against measuring two safety-filtered blanks against each other.

    Ideogram 4's filter rejects a prompt by returning a flat grey image, not by failing,
    so a comparison between two blocked outputs passes with a near-zero mismatch and
    proves nothing. Every quality number below is only meaningful if this passes.
    """
    energy = _edge_energy(fp8_baselines[case][0])
    assert energy > MIN_EDGE_ENERGY, (
        f"the fp8 baseline for {case!r} has edge energy {energy:.2f}, below {MIN_EDGE_ENERGY}: "
        "it is almost certainly a safety-filter blank, so any comparison against it is vacuous"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_fp8_generation_is_reproducible(case: str, fp8_baselines):
    """The zero point for every number this module reports.

    Each quantization figure is quoted as the cost of quantization, which is only true if
    generating twice from the *same* weights and seed lands in the same place. If this
    floor is non-zero, some share of those figures is run-to-run variance instead and has
    to be subtracted before any of them can be read as a quality claim.
    """
    first, second = fp8_baselines[case][0], fp8_baselines[case][1]
    mismatch = ImageCompare.check_images_close_enough(
        first,
        second,
        f"fp8 {case} is not reproducible across runs.",
        mismatch_threshold=NOISE_FLOOR_CEILING,
    )
    print(f"\nfp8 {case} noise floor: {mismatch:.2%} of subpixels differ between two identical runs")


def _require_checkpoint(bits: int) -> str:
    path = CHECKPOINTS[bits]
    env_name = f"MFLUX_IDEOGRAM4_Q{bits}_PATH"
    if not path:
        pytest.skip(f"{env_name} is not set; see this module's docstring")
    if not Path(path).exists():
        pytest.skip(f"{env_name} points at {path}, which does not exist")
    return path


@pytest.mark.slow
@pytest.mark.parametrize("bits", [8, 4])
def test_saved_checkpoint_declares_its_quantization_level(bits: int):
    """Every component must declare the level it was actually saved at.

    The bug that motivated this work: the transformers and the text encoder were marked
    skip_quantization, so `-q 4` wrote fp8 weights under a `quantization_level: 4` stamp.
    Metadata alone would not have caught that, which is why the generation tests below
    are the real assertion -- but a wrong stamp is worth failing fast on.
    """
    root = _require_checkpoint(bits)
    for component in ("transformer", "unconditional_transformer", "text_encoder", "vae"):
        index_path = Path(root) / component / "model.safetensors.index.json"
        assert index_path.exists(), f"{component} is missing from the saved checkpoint"
        metadata = json.loads(index_path.read_text()).get("metadata", {})
        assert metadata.get("quantization_level") == str(bits), (
            f"{component} declares quantization_level={metadata.get('quantization_level')!r}, expected {bits}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("case", sorted(CASES))
def test_quantized_output_matches_fp8(bits: int, case: str, fp8_baselines, quantized_outputs):
    _require_checkpoint(bits)
    assert bits in quantized_outputs, f"no output generated for q{bits}"

    ceiling = MISMATCH_CEILING[(case, bits)]
    mismatch = ImageCompare.check_images_close_enough(
        quantized_outputs[bits][case][0],
        fp8_baselines[case][0],
        f"q{bits} {case} diverges from the fp8 baseline.",
        mismatch_threshold=ceiling,
    )
    size = f"{CASES[case]['width']}x{CASES[case]['height']}"
    print(f"\nq{bits} {case} @ {size}: {mismatch:.2%} of subpixels differ (ceiling {ceiling:.0%})")
