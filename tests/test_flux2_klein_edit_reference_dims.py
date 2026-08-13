"""Klein edit reference-image conditioning dims (upstream #385).

Each reference image is conditioned at its OWN size (aspect-preserving downscale to at
most 1024x1024 pixels of area, then a center-crop down to multiples of 16), never
stretched to the output --width/--height.
"""

import PIL.Image
import pytest

from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers


@pytest.mark.fast
class TestKleinEditReferenceDims:
    def test_no_area_cap_pure_snap_down(self):
        # 717x403 (0.29MP, under the cap): only the snap-down to multiples of 16 applies.
        assert _Flux2KleinEditHelpers.reference_dims(717, 403) == (704, 400)

    def test_area_cap_aspect_preserving_downscale(self):
        # 1344x896 = 1204224 px (1.204MP) > 1048576 (the 1024*1024 cap):
        #   scale  = sqrt(1048576 / 1204224) = 0.9331389...
        #   width  = round(1344 * scale) = round(1254.14) = 1254
        #   height = round( 896 * scale) = round( 836.09) =  836
        # then snap down to multiples of 16:
        #   1254 - (1254 % 16) = 1254 - 6 = 1248
        #    836 - ( 836 % 16) =  836 - 4 =  832
        assert _Flux2KleinEditHelpers.reference_dims(1344, 896) == (1248, 832)

    def test_square_1024_unchanged(self):
        # Exactly at the area cap and already a multiple of 16: untouched.
        assert _Flux2KleinEditHelpers.reference_dims(1024, 1024) == (1024, 1024)

    def test_square_1000_snaps_down(self):
        assert _Flux2KleinEditHelpers.reference_dims(1000, 1000) == (992, 992)

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            (717, 403),
            (1344, 896),
            (1920, 1080),
            (640, 1026),
            (3023, 4031),
        ],
    )
    def test_aspect_ratio_preserved_within_3_percent(self, width, height):
        # The crop residual is < 16px per side, so the output aspect ratio stays
        # within 3% of the input's for any reasonably sized reference.
        out_w, out_h = _Flux2KleinEditHelpers.reference_dims(width, height)
        assert abs((out_w / out_h) / (width / height) - 1.0) < 0.03

    def test_too_small_reference_raises(self):
        with pytest.raises(ValueError, match="too small"):
            _Flux2KleinEditHelpers.reference_dims(10, 200)


@pytest.mark.fast
class TestKleinEditPrepareReferenceImage:
    def test_center_crop_loses_both_sides_equally(self):
        # Encode the pixel coordinates in the pixel values so the crop offsets are
        # recoverable from the output. 717x403 is under the area cap, so the result
        # must be a pure center-crop to 704x400 (13px of width and 3px of height lost).
        width, height = 717, 403
        image = PIL.Image.new("RGB", (width, height))
        image.putdata([(x % 256, y % 256, 0) for y in range(height) for x in range(width)])

        result = _Flux2KleinEditHelpers.prepare_reference_image(image)

        assert result.size == (704, 400)
        left_lost, top_lost, _ = result.getpixel((0, 0))
        right_lost = (width - 704) - left_lost
        bottom_lost = (height - 400) - top_lost
        assert abs(left_lost - right_lost) <= 1
        assert abs(top_lost - bottom_lost) <= 1

        # A true crop (no resampling): interior pixels map 1:1 to the original grid.
        assert result.getpixel((250, 250)) == ((left_lost + 250) % 256, (top_lost + 250) % 256, 0)
        assert result.getpixel((703, 399)) == ((left_lost + 703) % 256, (top_lost + 399) % 256, 0)
