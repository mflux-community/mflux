import numpy as np
import pytest

from mflux.models.openpose.openpose import OpenPoseBody


@pytest.mark.fast
def test_draw_renders_a_colored_skeleton_on_black():
    # One person whose first limb (keypoints 2-3 -> indices 1,2) connects two candidates. The renderer
    # should paint a colored limb + joints on an otherwise black canvas of the requested size.
    candidate = np.array([[30.0, 30.0, 0.9, 0], [30.0, 90.0, 0.9, 1]])
    person = -1 * np.ones(20)
    person[1], person[2] = 0, 1  # keypoint indices into `candidate`
    person[-1], person[-2] = 2, 1.8
    img = OpenPoseBody._draw((160, 120), candidate, np.array([person]))
    assert img.size == (120, 160)
    assert img.mode == "RGB"
    arr = np.array(img)
    assert arr.max() > 0  # something was drawn
    assert (arr == 0).mean() > 0.5  # skeleton on a mostly-black canvas


@pytest.mark.fast
def test_assemble_on_empty_maps_finds_no_people():
    # No activation anywhere -> no peaks, no limbs, no persons, and no crash.
    heat = np.zeros((64, 48, 19), dtype=np.float32)
    paf = np.zeros((64, 48, 38), dtype=np.float32)
    candidate, subset = OpenPoseBody._assemble(paf, heat)
    assert len(subset) == 0
