import json
from pathlib import Path

from PIL import Image

from mflux.utils.exceptions import PromptFileReadError
from mflux.utils.exif_orientation import open_oriented


class FiboVLMUtil:
    @staticmethod
    def save_json_prompt(output_path: Path, json_string: str) -> None:
        try:
            json_parsed = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"VLM did not return valid JSON: {e}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_parsed, f, indent=2, ensure_ascii=False)

    @staticmethod
    def get_structured_prompt(prompt_file: Path) -> str:
        if not prompt_file.exists():
            raise PromptFileReadError(f"Prompt file does not exist: {prompt_file}")
        with open(prompt_file, "rt") as f:
            structured_prompt = f.read().strip()
        if not structured_prompt:
            raise PromptFileReadError(f"Prompt file is empty: {prompt_file}")
        try:
            json.loads(structured_prompt)
        except json.JSONDecodeError as e:
            raise PromptFileReadError(f"Prompt file does not contain valid JSON: {e}")
        return structured_prompt

    @staticmethod
    def load_image(image_path: Path) -> Image.Image:
        if not image_path.exists():
            raise PromptFileReadError(f"Image file does not exist: {image_path}")
        try:
            image = open_oriented(image_path).convert("RGB")
        except (OSError, IOError, ValueError) as e:
            raise PromptFileReadError(f"Failed to load image: {e}")
        return image
