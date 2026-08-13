"""One reading of a file's EXIF Orientation, shared by everything that touches image paths.

Pixels and dimensions have to agree. Once a loader applies the tag, any other site that
reads the same file raw sees a picture rotated ninety degrees away from the one the model
is working on: a mask that no longer lines up, a target aspect that is inverted, a caption
describing a sideways scene.
"""

import PIL.Image
import PIL.ImageOps
from PIL._typing import StrOrBytesPath

# Orientation values whose transform swaps the axes; the rest keep width and height.
_ROTATING_ORIENTATIONS = frozenset({5, 6, 7, 8})
_ORIENTATION_TAG = 0x0112


def open_oriented(image_or_path: PIL.Image.Image | StrOrBytesPath) -> PIL.Image.Image:
    """Open an image with its Orientation tag applied, leaving the mode alone."""
    image = image_or_path if isinstance(image_or_path, PIL.Image.Image) else PIL.Image.open(image_or_path)
    return PIL.ImageOps.exif_transpose(image)


def oriented_size(path: StrOrBytesPath) -> tuple[int, int]:
    """The size the image displays at, without decoding its pixels.

    Callers that only need dimensions stay as cheap as a bare open: the tag is read from
    the header and the axes swapped, rather than materializing a transposed copy.
    """
    with PIL.Image.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(_ORIENTATION_TAG, 1)
    return (height, width) if orientation in _ROTATING_ORIENTATIONS else (width, height)
