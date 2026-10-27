import functools
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class Fibo2TextImages:
    # Each text_render text drawn as training drew it; missing glyphs come from Aileron, then DejaVu Sans
    FONT_SIZE = 20
    MAX_LINE_CHARS = 256
    PADDING = 2
    LINE_SPACING = 4
    FONT_FILES = ("Aileron-Regular.otf", "DejaVuSans.ttf")

    @staticmethod
    def draw_for_caption(caption: dict, fonts_dir: Path | None) -> list[Image.Image]:
        # One image per text_render entry that has text, in order
        fallback = Fibo2TextImages._load_fonts(fonts_dir)
        return [
            Fibo2TextImages.draw(entry["text"], fallback)
            for entry in caption.get("text_render") or []
            if Fibo2TextImages.has_text(entry)
        ]

    @staticmethod
    def has_text(entry) -> bool:
        return isinstance(entry, dict) and isinstance(entry.get("text"), str) and bool(entry["text"])

    @staticmethod
    def draw(text: str, fallback_fonts: list[ImageFont.FreeTypeFont]) -> Image.Image:
        # One line of collapsed whitespace, split into lines of 256 characters
        font = Fibo2TextImages._default_font()
        line = " ".join(text.split())
        size = Fibo2TextImages.MAX_LINE_CHARS
        lines = [line[i : i + size] for i in range(0, len(line), size)] or [""]
        if all(ord(character) < 128 or Fibo2TextImages._has_glyph(font, character) for character in line):
            return Fibo2TextImages._draw_one_font(lines, font)
        if not fallback_fonts:
            logger.warning("A text to draw has characters Pillow's built-in font lacks; FIBO 2's fonts were not found.")
        return Fibo2TextImages._draw_mixed(lines, [font, *fallback_fonts])

    @staticmethod
    def _load_fonts(fonts_dir: Path | None) -> list[ImageFont.FreeTypeFont]:
        if fonts_dir is None:
            return []
        paths = [Path(fonts_dir) / name for name in Fibo2TextImages.FONT_FILES]
        return [Fibo2TextImages._load_font(str(path)) for path in paths if path.is_file()]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _load_font(path: str) -> ImageFont.FreeTypeFont:
        # The basic layout, as Pillow's built-in font, so the text does not set differently where libraqm is installed
        return ImageFont.truetype(path, Fibo2TextImages.FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _default_font() -> ImageFont.FreeTypeFont:
        font = ImageFont.load_default(Fibo2TextImages.FONT_SIZE)
        if not isinstance(font, ImageFont.FreeTypeFont):
            raise ImportError("Drawing the images of a caption's text needs Pillow built with FreeType.")
        return font

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _has_glyph(font: ImageFont.FreeTypeFont, character: str) -> bool:
        # A font without a glyph for the character draws its "notdef" box, as it does for U+FFFE, which no font has
        try:
            notdef = font.getmask("￾")
            mask = font.getmask(character)
        except Exception:  # noqa: BLE001 - as in training, a font that can't say is taken to have the glyph
            return True
        return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))

    @staticmethod
    def _font_runs(text: str, fonts: list[ImageFont.FreeTypeFont]) -> list[tuple[str, ImageFont.FreeTypeFont]]:
        # Runs of characters sharing the first font that has their glyphs (the built-in font when none has)
        runs: list[list] = []
        for character in text:
            font = fonts[0]
            if not (ord(character) < 128 or Fibo2TextImages._has_glyph(fonts[0], character)):
                font = next((f for f in fonts[1:] if Fibo2TextImages._has_glyph(f, character)), fonts[0])
            if runs and runs[-1][1] is font:
                runs[-1][0] += character
            else:
                runs.append([character, font])
        return [(run, font) for run, font in runs]

    @staticmethod
    def _draw_one_font(lines: list[str], font: ImageFont.FreeTypeFont) -> Image.Image:
        # Each line is placed by its bounding box, and is at least as tall as "Ag"
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
        min_box = draw.textbbox((0, 0), "Ag", font=font)
        min_line_height = max(1, min_box[3] - min_box[1])
        line_sizes = []
        max_width = 1
        total_height = 0
        for line in lines:
            box = draw.textbbox((0, 0), line or " ", font=font)
            line_height = max(min_line_height, box[3] - box[1])
            line_sizes.append((box, line_height))
            max_width = max(max_width, box[2] - box[0])
            total_height += line_height
        total_height += Fibo2TextImages.LINE_SPACING * (len(lines) - 1)

        padding = Fibo2TextImages.PADDING
        image = Image.new("RGB", (max_width + padding * 2, total_height + padding * 2), "white")
        draw = ImageDraw.Draw(image)
        y = padding
        for line, (box, line_height) in zip(lines, line_sizes):
            draw.text((padding - box[0], y - box[1]), line, font=font, fill=(0, 0, 0))
            y += line_height + Fibo2TextImages.LINE_SPACING
        return image

    @staticmethod
    def _draw_mixed(lines: list[str], fonts: list[ImageFont.FreeTypeFont]) -> Image.Image:
        # Runs share the line's baseline; lines are as tall as the built-in font's ascent and descent
        ascent, descent = fonts[0].getmetrics()
        line_height = max(1, ascent + descent)
        runs_per_line = [Fibo2TextImages._font_runs(line or " ", fonts) for line in lines]
        width = max(max(1, round(sum(font.getlength(run) for run, font in runs))) for runs in runs_per_line)
        height = line_height * len(lines) + Fibo2TextImages.LINE_SPACING * (len(lines) - 1)

        padding = Fibo2TextImages.PADDING
        image = Image.new("RGB", (width + padding * 2, height + padding * 2), "white")
        draw = ImageDraw.Draw(image)
        y = padding
        for runs in runs_per_line:
            x = float(padding)
            for run, font in runs:
                draw.text((x, y + ascent), run, font=font, fill=(0, 0, 0), anchor="ls")
                x += font.getlength(run)
            y += line_height + Fibo2TextImages.LINE_SPACING
        return image
