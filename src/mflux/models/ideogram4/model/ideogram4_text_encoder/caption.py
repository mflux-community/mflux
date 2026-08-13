from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

NON_ASCII_UNICODE_ESCAPE_RE = re.compile(
    r"\\u(?:00[89a-fA-F][0-9a-fA-F]|0[1-9a-fA-F][0-9a-fA-F]{2}|[1-9a-fA-F][0-9a-fA-F]{3})"
)


class Ideogram4CaptionWarning(UserWarning):
    pass


@dataclass(frozen=True)
class Ideogram4PreparedPrompt:
    prompt: str
    warnings: tuple[str, ...]
    is_json_caption: bool


class Ideogram4CaptionVerifier:
    top_level_known_keys: frozenset[str] = frozenset(
        {
            "high_level_description",
            "style_description",
            "compositional_deconstruction",
        }
    )
    style_description_known_keys: frozenset[str] = frozenset(
        {
            "aesthetics",
            "lighting",
            "photo",
            "art_style",
            "medium",
            "color_palette",
        }
    )
    compositional_deconstruction_key_order: Sequence[str] = ("background", "elements")
    element_known_keys: frozenset[str] = frozenset(
        {
            "type",
            "bbox",
            "text",
            "desc",
            "color_palette",
        }
    )
    element_types: frozenset[str] = frozenset({"obj", "text"})

    def verify_raw(self, raw_text: str) -> list[str]:
        messages = self.check_ensure_ascii_false(raw_text)
        try:
            caption = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            messages.append(f"invalid JSON: {exc}")
            return messages
        return messages + self.verify(caption)

    def verify(self, caption: Any) -> list[str]:
        messages: list[str] = []
        if not isinstance(caption, dict):
            messages.append(f"root: expected a JSON object, got {type(caption).__name__}")
            return messages

        self._check_unknown_keys(caption, self.top_level_known_keys, "root", messages)
        if "high_level_description" in caption and not isinstance(caption["high_level_description"], str):
            messages.append(
                f"high_level_description: expected a string, got {type(caption['high_level_description']).__name__}"
            )
        if "style_description" in caption:
            self._verify_style_description(caption["style_description"], messages)
        if "compositional_deconstruction" in caption:
            self._verify_compositional_deconstruction(caption["compositional_deconstruction"], messages)
        else:
            messages.append("root: 'compositional_deconstruction' should exist")
        return messages

    @classmethod
    def check_ensure_ascii_false(cls, raw_text: str) -> list[str]:
        matches = NON_ASCII_UNICODE_ESCAPE_RE.findall(raw_text)
        if not matches or any(ord(char) > 0x7F for char in raw_text):
            return []
        examples = ", ".join(sorted(set(matches))[:3])
        return [
            "raw text: found non-ASCII unicode escapes "
            f"({examples}); use json.dumps(..., ensure_ascii=False) for JSON captions"
        ]

    def _verify_style_description(self, style_description: Any, messages: list[str]) -> None:
        if not isinstance(style_description, dict):
            messages.append(f"style_description: expected an object, got {type(style_description).__name__}")
            return

        self._check_unknown_keys(
            style_description,
            self.style_description_known_keys,
            "style_description",
            messages,
        )
        has_photo = "photo" in style_description
        has_art_style = "art_style" in style_description
        if has_photo == has_art_style:
            messages.append("style_description: expected exactly one of 'photo' or 'art_style'")
            return

        expected_order = self._style_description_key_order(style_description)
        self._check_key_order(style_description, expected_order, "style_description", messages)
        for key in expected_order:
            if key == "color_palette":
                continue
            if key not in style_description:
                messages.append(f"style_description: '{key}' should exist")
            elif not isinstance(style_description[key], str):
                messages.append(f"style_description.{key}: expected a string")

        if "color_palette" in style_description:
            self._verify_color_palette(
                style_description["color_palette"],
                "style_description.color_palette",
                max_colors=16,
                messages=messages,
            )

    def _verify_compositional_deconstruction(self, value: Any, messages: list[str]) -> None:
        if not isinstance(value, dict):
            messages.append(f"compositional_deconstruction: expected an object, got {type(value).__name__}")
            return

        self._check_key_order(
            value,
            self.compositional_deconstruction_key_order,
            "compositional_deconstruction",
            messages,
        )
        if "background" not in value:
            messages.append("compositional_deconstruction: 'background' should exist")
        elif not isinstance(value["background"], str):
            messages.append("compositional_deconstruction.background: expected a string")

        if "elements" not in value:
            messages.append("compositional_deconstruction: 'elements' should exist")
            return
        if not isinstance(value["elements"], list):
            messages.append("compositional_deconstruction.elements: expected a list")
            return
        for index, element in enumerate(value["elements"]):
            self._verify_element(index, element, messages)

    def _verify_element(self, index: int, element: Any, messages: list[str]) -> None:
        path = f"elements[{index}]"
        if not isinstance(element, dict):
            messages.append(f"{path}: expected an object, got {type(element).__name__}")
            return

        self._check_unknown_keys(element, self.element_known_keys, path, messages)
        element_type = element.get("type")
        if element_type not in self.element_types:
            messages.append(f"{path}: 'type' should be one of {sorted(self.element_types)}")
            return

        self._check_key_order(element, self._element_key_order(element), path, messages)
        if "desc" not in element:
            messages.append(f"{path}: 'desc' should exist")
        elif not isinstance(element["desc"], str):
            messages.append(f"{path}.desc: expected a string")

        if element_type == "text":
            if "text" not in element:
                messages.append(f"{path}: text elements should include 'text'")
            elif not isinstance(element["text"], str):
                messages.append(f"{path}.text: expected a string")

        if "bbox" in element:
            self._verify_bbox(index, element["bbox"], messages)
        if "color_palette" in element:
            self._verify_color_palette(
                element["color_palette"],
                f"{path}.color_palette",
                max_colors=5,
                messages=messages,
            )

    @staticmethod
    def _style_description_key_order(style_description: dict[str, Any]) -> Sequence[str]:
        if "photo" in style_description:
            order = ("aesthetics", "lighting", "photo", "medium", "color_palette")
        else:
            order = ("aesthetics", "lighting", "medium", "art_style", "color_palette")
        return tuple(key for key in order if key != "color_palette" or key in style_description)

    @staticmethod
    def _element_key_order(element: dict[str, Any]) -> Sequence[str]:
        if element.get("type") == "text":
            order = ("type", "bbox", "text", "desc", "color_palette")
        else:
            order = ("type", "bbox", "desc", "color_palette")
        return tuple(key for key in order if key not in {"bbox", "color_palette"} or key in element)

    @staticmethod
    def _verify_bbox(index: int, bbox: Any, messages: list[str]) -> None:
        path = f"elements[{index}].bbox"
        if not isinstance(bbox, list) or len(bbox) != 4:
            messages.append(f"{path}: expected [y_min, x_min, y_max, x_max]")
            return
        if not all(type(value) is int for value in bbox):
            messages.append(f"{path}: all values should be integers")
            return
        y_min, x_min, y_max, x_max = bbox
        if not all(0 <= value <= 1000 for value in bbox):
            messages.append(f"{path}: values should be in [0, 1000], got {bbox}")
        if y_min > y_max:
            messages.append(f"{path}: y_min ({y_min}) should be <= y_max ({y_max})")
        if x_min > x_max:
            messages.append(f"{path}: x_min ({x_min}) should be <= x_max ({x_max})")

    @staticmethod
    def _verify_color_palette(palette: Any, path: str, max_colors: int, messages: list[str]) -> None:
        if not isinstance(palette, list):
            messages.append(f"{path}: expected a list")
            return
        if len(palette) > max_colors:
            messages.append(f"{path}: expected at most {max_colors} colors, got {len(palette)}")
        for index, color in enumerate(palette):
            if (
                not isinstance(color, str)
                or len(color) != 7
                or not color.startswith("#")
                or not all(char in "0123456789ABCDEF" for char in color[1:])
            ):
                messages.append(f"{path}[{index}]: expected uppercase #RRGGBB hex color")

    @staticmethod
    def _check_key_order(
        obj: dict[str, Any],
        expected_order: Sequence[str],
        path: str,
        messages: list[str],
    ) -> None:
        present_keys = tuple(key for key in obj if key in expected_order)
        if present_keys != tuple(expected_order):
            messages.append(f"{path}: key order is {present_keys}, expected {tuple(expected_order)}")
        extra_keys = [key for key in obj if key not in expected_order]
        if extra_keys:
            messages.append(f"{path}: keys {extra_keys} are not allowed in this context")

    @staticmethod
    def _check_unknown_keys(
        obj: dict[str, Any],
        known_keys: frozenset[str],
        path: str,
        messages: list[str],
    ) -> None:
        unknown_keys = [key for key in obj if key not in known_keys]
        if unknown_keys:
            messages.append(f"{path}: unknown keys {unknown_keys}")


class Ideogram4Caption:
    PLAIN_PROMPT_WARNING = (
        "plain prompt: Ideogram 4 was trained on structured JSON captions; plain text may reduce quality "
        "and increase safety-filter false positives"
    )

    @staticmethod
    def prepare(prompt: str | dict[str, Any]) -> Ideogram4PreparedPrompt:
        verifier = Ideogram4CaptionVerifier()
        if isinstance(prompt, dict):
            normalized = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
            return Ideogram4PreparedPrompt(
                prompt=normalized,
                warnings=tuple(verifier.verify(prompt)),
                is_json_caption=True,
            )
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string or JSON-caption dict, got {type(prompt).__name__}")

        stripped = prompt.strip()
        if not stripped.startswith("{"):
            return Ideogram4PreparedPrompt(
                prompt=prompt,
                warnings=(Ideogram4Caption.PLAIN_PROMPT_WARNING,),
                is_json_caption=False,
            )

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return Ideogram4PreparedPrompt(
                prompt=prompt,
                warnings=(f"invalid JSON caption: {exc}; prompt will be used verbatim",),
                is_json_caption=False,
            )
        if not isinstance(parsed, dict):
            return Ideogram4PreparedPrompt(
                prompt=prompt,
                warnings=(
                    f"JSON caption should be an object, got {type(parsed).__name__}; prompt will be used verbatim",
                ),
                is_json_caption=False,
            )

        normalized = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return Ideogram4PreparedPrompt(
            prompt=normalized,
            warnings=tuple(verifier.verify_raw(stripped)),
            is_json_caption=True,
        )

    @staticmethod
    def raise_for_warnings(messages: Sequence[str]) -> None:
        if messages:
            formatted = "\n".join(f"- {message}" for message in messages)
            raise ValueError(f"Ideogram 4 caption validation failed:\n{formatted}")

    @staticmethod
    def emit_warnings(messages: Sequence[str], stacklevel: int = 2) -> None:
        for message in messages:
            warnings.warn(
                f"Ideogram 4 caption: {message}",
                Ideogram4CaptionWarning,
                stacklevel=stacklevel,
            )
