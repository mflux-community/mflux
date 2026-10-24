import pytest
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo2.variants.edit.fibo2_edit_images import Fibo2EditImages

pytestmark = pytest.mark.fast

# The (width, height) diffusers' BriaFibo2ImageProcessor.get_image_size gives for these aspect ratios
TRAINED_SIZES = {
    1.0: (1024, 1024),
    4 / 3: (1152, 864),
    3 / 4: (864, 1152),
    16 / 9: (1424, 800),
    9 / 16: (800, 1424),
    3 / 2: (1248, 832),
    2.35: (1504, 640),
    0.5: (720, 1440),
}


class _Images:
    @staticmethod
    def save(path, size, color=(100, 150, 200), mode="RGB"):
        Image.new(mode, size, color).save(path)
        return path


@pytest.mark.parametrize(("aspect_ratio", "size"), TRAINED_SIZES.items())
def test_trained_size_matches_diffusers(aspect_ratio, size):
    assert Fibo2EditImages.trained_size(aspect_ratio) == size


def test_one_image_without_a_size_takes_the_trained_size_for_its_aspect_ratio(tmp_path):
    crops, copies, size = Fibo2EditImages.prepare([_Images.save(tmp_path / "wide.png", (1600, 900))])
    assert size == (1424, 800)
    assert crops[0].size == (1424, 800)
    assert max(copies[0].size) == Fibo2EditImages.TEXT_ENCODER_IMAGE_SIZE


def test_several_images_each_take_the_trained_size_for_their_own_aspect_ratio(tmp_path):
    paths = [_Images.save(tmp_path / "square.png", (500, 500)), _Images.save(tmp_path / "wide.png", (1200, 800))]
    crops, _, size = Fibo2EditImages.prepare(paths)
    assert size == (1024, 1024)
    assert [crop.size for crop in crops] == [(1024, 1024), (1248, 832)]


@pytest.mark.parametrize(("width", "height"), [(1000, 700), (1001, 777), (1010, 1032), (1008, 1008)])
def test_an_explicit_size_crops_to_the_size_of_the_generated_latents(tmp_path, width, height):
    crops, _, size = Fibo2EditImages.prepare([_Images.save(tmp_path / "a.png", (800, 600))], width=width, height=height)
    config = Config(model_config=ModelConfig.fibo2(), num_inference_steps=1, width=width, height=height)
    assert size == (config.width, config.height)
    assert crops[0].size == (config.width, config.height)


@pytest.mark.parametrize(("width", "height", "size"), [(768, None, (768, 512)), (None, 512, (768, 512)), (1000, None, (992, 656))])  # fmt: off
def test_one_side_given_keeps_the_aspect_ratio_of_the_first_image(tmp_path, width, height, size):
    # The other side was 1024, so a 3:2 photo edited at --width 768 came out cropped to 768x1024
    crops, _, result = Fibo2EditImages.prepare([_Images.save(tmp_path / "a.png", (1200, 800))], width=width, height=height)  # fmt: off
    assert result == size
    assert crops[0].size == size


def test_exif_orientation_is_applied_to_the_image(tmp_path):
    # A landscape file whose Orientation tag says to show it rotated: the model must see it upright (portrait)
    path = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (1200, 900), (10, 200, 30)).save(path, exif=exif)
    crops, _, size = Fibo2EditImages.prepare([path])
    assert size == Fibo2EditImages.trained_size(900 / 1200)
    assert crops[0].height > crops[0].width


def test_a_transparent_image_is_composited_over_white(tmp_path):
    # mflux's image loader composites transparency over white; diffusers converts to RGB, which shows whatever
    # colour sits under the transparent pixels (often black)
    path = _Images.save(tmp_path / "cutout.png", (64, 64), (255, 0, 0, 0), mode="RGBA")
    crops, _, _ = Fibo2EditImages.prepare([path], width=64, height=64)
    assert crops[0].getpixel((0, 0)) == (255, 255, 255)


def test_the_mask_greys_out_its_region_at_128():
    image = Image.new("RGB", (64, 64), (255, 0, 0))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(128, (0, 0, 32, 64))
    mask.paste(127, (32, 0, 64, 64))
    greyed = Fibo2EditImages.grey_out(image, mask)
    assert greyed.getpixel((10, 10)) == (128, 128, 128)
    assert greyed.getpixel((50, 10)) == (255, 0, 0)


def test_a_mask_of_another_size_is_refused(tmp_path):
    image = _Images.save(tmp_path / "a.png", (64, 64))
    mask = _Images.save(tmp_path / "mask.png", (32, 32), 255, mode="L")
    with pytest.raises(ValueError, match="mask has size"):
        Fibo2EditImages.prepare([image], mask_path=mask)


def test_image_count_and_mask_rules(tmp_path):
    image = _Images.save(tmp_path / "a.png", (64, 64))
    with pytest.raises(ValueError, match="1 to 5"):
        Fibo2EditImages.prepare([image] * 6)
    with pytest.raises(ValueError, match="single image"):
        Fibo2EditImages.prepare([image, image], mask_path=image)
