import difflib
import json
import logging
import math
import re
from argparse import Namespace
from pathlib import Path

from mflux.models.fibo2.model.fibo2_text_encoder.prompt_encoder import Fibo2PromptEncoder

logger = logging.getLogger(__name__)


class Fibo2CaptionCheck:
    # A JSON caption against the format FIBO 2 was trained on: ThalesVGL in bria-vgl 0.10.0 (vgl/README.md in
    # briaai/Fibo-2.0-dev). A field of the wrong type is an error. An unknown field or a missing required one is a
    # warning, since the reference pipeline passes such a caption to the model as it is. The check runs on the caption
    # as the pipeline cleans it, so empty values and FIBO 1's scores are not reported.
    STRING, INTEGER = "a string", "an integer"
    _INTEGER_TEXT = re.compile(r"^\s*[+-]?\d[\d_]*(\.0*)?\s*$")
    OBJECT = {
        "description": (STRING, True),
        "location": (STRING, True),
        "relationship": (STRING, True),
        "relative_size": (STRING, False),
        "shape_and_color": (STRING, False),
        "texture": (STRING, False),
        "appearance_details": (STRING, False),
        "number_of_objects": (INTEGER, False),
        "pose": (STRING, False),
        "expression": (STRING, False),
        "clothing": (STRING, False),
        "action": (STRING, False),
        "gender": (STRING, False),
        "skin_tone_and_texture": (STRING, False),
        "orientation": (STRING, False),
    }
    LIGHTING = {"conditions": (STRING, True), "direction": (STRING, True), "shadows": (STRING, False)}
    AESTHETICS = {"composition": (STRING, True), "color_scheme": (STRING, True), "mood_atmosphere": (STRING, True)}
    PHOTOGRAPHIC = {
        "depth_of_field": (STRING, True),
        "focus": (STRING, True),
        "camera_angle": (STRING, True),
        "lens_focal_length": (STRING, True),
    }
    TEXT_RENDER = {
        "text": (STRING, True),
        "location": (STRING, True),
        "size": (STRING, True),
        "color": (STRING, True),
        "font": (STRING, True),
        "appearance_details": (STRING, False),
    }
    CAPTION = {
        "short_description": (STRING, True),
        "objects": ([OBJECT], False),
        "background_setting": (STRING, True),
        "lighting": (LIGHTING, True),
        "aesthetics": (AESTHETICS, True),
        "photographic_characteristics": (PHOTOGRAPHIC, False),
        "style_medium": (STRING, False),
        "text_render": ([TEXT_RENDER], False),
        "context": (STRING, True),
        "artistic_style": (STRING, False),
    }

    @staticmethod
    def check(prompt: str, edit: bool) -> list[str]:
        # Raises ValueError on an error and returns the warnings. A prompt that is not a JSON object is plain text and
        # passes, unless it starts like a JSON object and does not parse.
        stripped = prompt.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if stripped.startswith("{"):
                raise ValueError(
                    f"The prompt starts like a JSON caption but is not valid JSON: {exc.msg} at line {exc.lineno}, "
                    f"column {exc.colno}. Fix the JSON, or give a plain-text prompt."
                ) from exc
            return []
        if not isinstance(payload, dict):
            return []
        warnings: list[str] = []
        if edit:
            Fibo2PromptEncoder.check_edit_prompt(stripped)
            instruction = payload["edit_instruction"]
            if not isinstance(instruction, str):
                raise ValueError(f'"edit_instruction" has to be a string, not {Fibo2CaptionCheck._kind(instruction)}.')
            if not instruction.strip():
                warnings.append("edit_instruction is empty.")
        elif "edit_instruction" in payload:
            warnings.append("edit_instruction is read only when editing images; this generation ignores it.")
        caption, _ = Fibo2PromptEncoder._parse_edit_prompt(stripped)
        if not isinstance(caption, dict):
            return warnings
        errors: list[str] = []
        Fibo2CaptionCheck._check_fields(caption, Fibo2CaptionCheck.CAPTION, "", errors, warnings)
        if errors:
            raise ValueError("The JSON caption does not have the format FIBO 2 reads:\n  " + "\n  ".join(errors))
        return warnings

    @staticmethod
    def check_and_log(prompt: str, edit: bool) -> None:
        # For generate_image: an error raises, each warning is logged
        for message in Fibo2CaptionCheck.check(prompt, edit=edit):
            logger.warning(message)

    @staticmethod
    def check_args(args: Namespace, edit: bool) -> None:
        # For a command, before the model loads: an error stops it. stdin ("-") is read once, at generation, and
        # checked there; a prompt file that cannot be read is reported by the generation loop.
        if args.prompt == "-":
            return
        prompt = args.prompt
        if args.prompt_file is not None:
            if not Path(args.prompt_file).is_file():
                return
            prompt = Path(args.prompt_file).read_text()
        if prompt:
            Fibo2CaptionCheck.check(prompt, edit=edit)

    @staticmethod
    def _check_fields(value: dict, spec: dict, path: str, errors: list[str], warnings: list[str]) -> None:
        missing = [name for name, (_, required) in spec.items() if required and name not in value]
        if missing:
            where = f" in {path}" if path else ""
            warnings.append(f"Missing{where}: {', '.join(missing)}. FIBO 2's training captions have these fields.")
        for name, item in value.items():
            if name not in spec:
                close = difflib.get_close_matches(name, list(spec), n=1)
                hint = f" Did you mean {close[0]!r}?" if close else ""
                warnings.append(f"Unknown field {Fibo2CaptionCheck._join(path, name)!r}: the model was not trained on it.{hint}")  # fmt: off
                continue
            Fibo2CaptionCheck._check_value(item, spec[name][0], Fibo2CaptionCheck._join(path, name), errors, warnings)

    @staticmethod
    def _check_value(value, expected, path: str, errors: list[str], warnings: list[str]) -> None:
        if expected == Fibo2CaptionCheck.STRING:
            if not isinstance(value, str):
                errors.append(f"{path} has to be a string, not {Fibo2CaptionCheck._kind(value)}.")
        elif expected == Fibo2CaptionCheck.INTEGER:
            if not Fibo2CaptionCheck._is_integer(value):
                errors.append(f"{path} has to be an integer, not {Fibo2CaptionCheck._kind(value)}.")
        elif isinstance(expected, list):
            if not isinstance(value, list):
                errors.append(f"{path} has to be a list of objects, not {Fibo2CaptionCheck._kind(value)}.")
                return
            for index, entry in enumerate(value):
                Fibo2CaptionCheck._check_value(entry, expected[0], f"{path}[{index}]", errors, warnings)
        elif not isinstance(value, dict):
            errors.append(f"{path} has to be a JSON object, not {Fibo2CaptionCheck._kind(value)}.")
        else:
            Fibo2CaptionCheck._check_fields(value, expected, path, errors, warnings)

    @staticmethod
    def _is_integer(value) -> bool:
        # What ThalesVGL (pydantic, lax mode) takes as an integer: an int or a boolean, an integral float, or a string
        # that writes an integer
        if isinstance(value, int):
            return True
        if isinstance(value, float):
            return math.isfinite(value) and value.is_integer()
        return isinstance(value, str) and Fibo2CaptionCheck._INTEGER_TEXT.match(value) is not None

    @staticmethod
    def _join(path: str, name: str) -> str:
        return f"{path}.{name}" if path else name

    @staticmethod
    def _kind(value) -> str:
        if isinstance(value, bool):
            return f"the boolean {json.dumps(value)}"
        if isinstance(value, (int, float)):
            return f"the number {value!r}"
        if isinstance(value, str):
            return f"the string {value!r}"
        return "a list" if isinstance(value, list) else "a JSON object"
