import math
from pathlib import Path

from PIL import Image

from mflux.utils.image_util import ImageUtil


class Fibo2EditImages:
    # The images to edit, prepared as FIBO 2 was trained on them
    MAX_IMAGES = 5
    AREA = 1024 * 1024
    MULTIPLE = 16
    TEXT_ENCODER_IMAGE_SIZE = 768

    @staticmethod
    def prepare(
        image_paths: list[Path | str],
        width: int | None = None,
        height: int | None = None,
        mask_path: Path | str | None = None,
    ) -> tuple[list[Image.Image], list[Image.Image], tuple[int, int]]:
        # One image is cropped to the result's size, several each to their own trained size
        if not 1 <= len(image_paths) <= Fibo2EditImages.MAX_IMAGES:
            raise ValueError(f"FIBO 2 edits 1 to {Fibo2EditImages.MAX_IMAGES} images, got {len(image_paths)}.")
        if mask_path is not None and len(image_paths) != 1:
            raise ValueError("A mask can only be used with a single image.")
        images = [ImageUtil.load_image(path) for path in image_paths]
        if mask_path is not None:
            mask = ImageUtil.load_image(mask_path, composite_alpha=False)
            if mask.size != images[0].size:
                raise ValueError(f"The mask has size {mask.size} but the image has size {images[0].size}.")
            images = [Fibo2EditImages.grey_out(images[0], mask)]
        first = images[0]
        width, height = Fibo2EditImages._result_size(first.width / first.height, width, height)
        # Rounded down to a multiple of 16, as the generated latents are
        width, height = 16 * (width // 16), 16 * (height // 16)
        if len(images) == 1:
            sizes = [(width, height)]
        else:
            sizes = [Fibo2EditImages.trained_size(image.width / image.height) for image in images]
        crops = [Fibo2EditImages._resize_and_crop(image, w, h) for image, (w, h) in zip(images, sizes)]
        text_encoder_images = []
        for crop in crops:
            copy = crop.copy()
            copy.thumbnail((Fibo2EditImages.TEXT_ENCODER_IMAGE_SIZE, Fibo2EditImages.TEXT_ENCODER_IMAGE_SIZE))
            text_encoder_images.append(copy)
        return crops, text_encoder_images, (width, height)

    @staticmethod
    def trained_size(aspect_ratio: float) -> tuple[int, int]:
        # Of the sizes near the area, the one that crops least, then closest in area, then the wider
        multiple, area = Fibo2EditImages.MULTIPLE, Fibo2EditImages.AREA
        min_side = 4 * multiple
        sizes = [
            (width, height)
            for width in range(min_side, 2 * area // min_side, multiple)
            for height in range(min_side, 2 * area // width, multiple)
            if 0.9 * area <= width * height <= 1.1 * area
        ]
        ranked = [
            (round(Fibo2EditImages._crop(size, aspect_ratio), 9), abs(size[0] * size[1] - area), -size[0], size)
            for size in sizes
        ]
        return min(ranked)[3]

    @staticmethod
    def _result_size(aspect_ratio: float, width: int | None, height: int | None) -> tuple[int, int]:
        # Without width and height, the trained size for the first image's aspect ratio; with one, the other keeps it
        if width is not None and height is not None:
            return width, height
        if width is not None:
            return width, round(width / aspect_ratio)
        if height is not None:
            return round(height * aspect_ratio), height
        return Fibo2EditImages.trained_size(aspect_ratio)

    @staticmethod
    def grey_out(image: Image.Image, mask: Image.Image) -> Image.Image:
        # Training greyed the masked region out flat, through the mask made binary at 128
        grey = Image.new("RGB", image.size, (128, 128, 128))
        binary_mask = mask.convert("L").point([0] * 128 + [255] * 128)
        return Image.composite(grey, image, binary_mask)

    @staticmethod
    def _resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
        # Scale to cover width x height keeping the aspect ratio, then crop the center
        scale = max(height / image.height, width / image.width)
        resized_width, resized_height = math.ceil(image.width * scale), math.ceil(image.height * scale)
        image = image.resize((resized_width, resized_height), resample=Image.LANCZOS)
        left, top = (resized_width - width) // 2, (resized_height - height) // 2
        return image.crop((left, top, left + width, top + height))

    @staticmethod
    def _crop(size: tuple[int, int], aspect_ratio: float) -> float:
        ratio = size[0] / size[1]
        return 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)
