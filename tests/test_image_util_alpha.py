import PIL.Image
import pytest

from mflux.utils.image_util import ImageUtil


def _rgba_with_hidden_rgb(hidden_rgb: tuple[int, int, int]) -> PIL.Image.Image:
    # Every pixel is fully transparent except one opaque pixel, so the visible
    # result must not depend on the RGB stored under the transparent field.
    image = PIL.Image.new("RGBA", (8, 8), (*hidden_rgb, 0))
    image.putpixel((4, 4), (200, 60, 30, 255))
    return image


@pytest.mark.fast
def test_load_image_composites_alpha_over_opaque_white():
    # convert("RGB") on an RGBA image drops alpha and exposes whatever RGB lives
    # under fully transparent pixels; that hidden payload must never reach the model.
    loaded = ImageUtil.load_image(_rgba_with_hidden_rgb((0, 0, 0)))

    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((4, 4)) == (200, 60, 30)


@pytest.mark.fast
def test_load_image_is_independent_of_hidden_rgb():
    # Two files that look identical on screen must load to the same array.
    black = ImageUtil.load_image(_rgba_with_hidden_rgb((0, 0, 0)))
    red = ImageUtil.load_image(_rgba_with_hidden_rgb((240, 30, 10)))

    assert black.tobytes() == red.tobytes()


@pytest.mark.fast
def test_load_image_keeps_opaque_rgb_unchanged():
    image = PIL.Image.new("RGB", (4, 4), (10, 20, 30))

    assert ImageUtil.load_image(image).getpixel((0, 0)) == (10, 20, 30)


@pytest.mark.fast
def test_load_image_composites_la_and_palette_transparency():
    la = PIL.Image.new("LA", (4, 4), (0, 0))
    la.putpixel((1, 1), (255, 255))
    assert ImageUtil.load_image(la).getpixel((0, 0)) == (255, 255, 255)

    palette = PIL.Image.new("P", (8, 8))
    palette.putpalette([0, 0, 0, 200, 60, 30] + [0, 0, 0] * 254)
    palette.info["transparency"] = 0
    for y in range(2, 6):
        for x in range(2, 6):
            palette.putpixel((x, y), 1)
    loaded = ImageUtil.load_image(palette)
    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((3, 3)) == (200, 60, 30)


@pytest.mark.fast
def test_load_image_blends_translucent_edges_over_white():
    image = PIL.Image.new("RGBA", (2, 1), (0, 0, 0, 128))
    image.putpixel((1, 0), (255, 0, 0, 255))

    loaded = ImageUtil.load_image(image)

    # A half-transparent black pixel over white is mid-grey, not black.
    assert loaded.getpixel((0, 0)) == (127, 127, 127)


@pytest.mark.fast
def test_load_image_accepts_a_path(tmp_path):
    path = tmp_path / "cutout.png"
    _rgba_with_hidden_rgb((0, 0, 0)).save(path, format="PNG")

    loaded = ImageUtil.load_image(path)

    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((4, 4)) == (200, 60, 30)


@pytest.mark.fast
def test_load_image_keeps_mask_transparency_unchanged_when_compositing_is_off():
    # A mask drawn as white strokes on a transparent background must keep its
    # transparent area black ("keep"), not white ("inpaint").
    mask = PIL.Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    mask.putpixel((1, 1), (255, 255, 255, 255))

    loaded = ImageUtil.load_image(mask, composite_alpha=False)

    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (0, 0, 0)
    assert loaded.getpixel((1, 1)) == (255, 255, 255)


@pytest.mark.fast
@pytest.mark.parametrize("mode", ["PA", "RGBa", "La"])
def test_load_image_composites_other_alpha_modes(mode):
    # Build "La" directly: Pillow's RGBA -> La conversion sets alpha to 255.
    if mode == "La":
        image = PIL.Image.new("La", (4, 4), (0, 0))
    else:
        image = PIL.Image.new("RGBA", (4, 4), (0, 0, 0, 0)).convert(mode)

    assert ImageUtil.load_image(image).getpixel((0, 0)) == (255, 255, 255)
