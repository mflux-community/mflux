import json
import re

import numpy as np
from PIL import Image, ImageFilter

# Qwen3-VL grounding: the model answers a locate request with a JSON bounding box. The
# encoder ships the full language model with its own lm_head, so the edit variant can
# ask it where an object is and turn the answer into an inpaint mask without extra weights.
GROUNDING_PROMPT = "Outline the position of {query} and output the bbox coordinates in JSON format."

# Condensation of the official Qwen-Image-2.1 prompt-rewrite recipe
# (QwenLM/Qwen-Image-2.1 prompt_rewrite/prompts/system_prompt_edit.txt, v2): rewrite a
# terse edit instruction into one detailed descriptive paragraph, keep the edit's
# language for the prose, keep any quoted text to be painted into the image verbatim,
# describe what must stay unchanged, and answer as JSON with a rewritten_prompt field.
REWRITE_PROMPT = (
    "You rewrite terse image-editing instructions into detailed prompts for an image "
    "editing model. Look at <image1>, the image to be edited. Rewrite the instruction "
    "below into ONE descriptive paragraph of 40-120 words that:\n"
    "- describes the requested change concretely (color, material, style, position) as "
    "already applied to the image;\n"
    "- explicitly states what must stay unchanged (identity, pose, background, "
    "lighting);\n"
    "- writes the description in the same language as the instruction; any text that "
    "should appear IN the image goes in double quotes, verbatim;\n"
    "- drops meta-commentary, options, and questions.\n"
    'Answer ONLY with JSON: {{"rewritten_prompt": "..."}}\n\nInstruction: {instruction}'
)

# Post-edit verification: the original and the edited output are shown together; the
# model answers whether the instruction was applied and everything else was preserved.
VERIFY_PROMPT = (
    "<image1> is the original image and <image2> is an edited version of it. The edit "
    'instruction was: "{instruction}". Compare them and answer ONLY with JSON: '
    '{{"instruction_applied": true or false, "outside_unchanged": true or false}}. '
    "instruction_applied means the requested change is clearly visible in <image2>; "
    "outside_unchanged means everything not targeted by the instruction (subject "
    "identity, pose, background composition) is preserved."
)

_BBOX_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


class QwenImage21Grounding:
    GROUNDING_PROMPT = GROUNDING_PROMPT
    REWRITE_PROMPT = REWRITE_PROMPT
    VERIFY_PROMPT = VERIFY_PROMPT

    @staticmethod
    def chat(instruction: str, image_count: int) -> str:
        # Chat-shaped request over image_count images; the processor expands each
        # <|image_pad|> placeholder to that image's merged token count.
        images = "<|vision_start|><|image_pad|><|vision_end|>" * image_count
        return f"<|im_start|>user\n{images}{instruction}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def balanced_json_spans(text: str) -> list[str]:
        # Substrings that hold a balanced {...} span, in order of appearance (ported
        # from the official prompt_rewrite answer parser).
        spans, depth, start = [], 0, -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : i + 1])
        return spans

    @staticmethod
    def parse_rewrite(text: str) -> str | None:
        # The rewrite answer's JSON carries {"rewritten_prompt": "..."} (the official
        # parser also tolerates a "rewrited_prompt" typo); scan last-first like the
        # official parser since the object is emitted at the end.
        for candidate in reversed(QwenImage21Grounding.balanced_json_spans(text)):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
            if isinstance(rewritten, str) and rewritten.strip():
                return rewritten.strip()
        return None

    @staticmethod
    def parse_verification(text: str) -> tuple[bool, bool] | None:
        # Returns (instruction_applied, outside_unchanged), or None when the reply
        # carries no parseable verdict.
        applied = unchanged = None
        for candidate in reversed(QwenImage21Grounding.balanced_json_spans(text)):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if "instruction_applied" in obj:
                applied = bool(obj["instruction_applied"])
                unchanged = bool(obj.get("outside_unchanged", True))
                return applied, unchanged
        # tolerate a plain-language reply
        lowered = text.lower()
        if "instruction_applied" in lowered:
            applied = re.search(r'applied"?\s*:\s*true', lowered) is not None
            unchanged = re.search(r'unchanged"?\s*:\s*true', lowered) is not None
            return applied, unchanged
        return None

    @staticmethod
    def parse_bbox(text: str) -> tuple[float, float, float, float] | None:
        # Returns (x1, y1, x2, y2) as fractions of the shown image, or None when the reply
        # carries no plausible box. Qwen3-VL grounds in 0-1000 coordinates relative to the
        # shown image, whatever its pixel size.
        for match in _BBOX_PATTERN.finditer(text):
            x1, y1, x2, y2 = (float(v) / 1000.0 for v in match.groups())
            if x2 <= x1 or y2 <= y1 or max(x1, y1, x2, y2) > 1.5:
                continue
            clamped = tuple(min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
            if (clamped[2] - clamped[0]) * (clamped[3] - clamped[1]) < 1e-4:
                continue
            return clamped
        return None

    @staticmethod
    def rasterize_mask(
        bbox: tuple[float, float, float, float], size: tuple[int, int], feather_fraction: float = 0.01
    ) -> Image.Image:
        # The bounding box as a soft binary mask over the output image: white where the
        # model should repaint, blurred slightly so the latent blend has no hard seam.
        # The box is grown by 3% per side first: grounding boxes tend to hug the object
        # tightly, and an uncovered sliver would keep its original color after inpainting.
        width, height = size
        x1, y1, x2, y2 = bbox
        grow_x, grow_y = 0.03 * (x2 - x1), 0.03 * (y2 - y1)
        x1, x2 = max(0.0, x1 - grow_x), min(1.0, x2 + grow_x)
        y1, y2 = max(0.0, y1 - grow_y), min(1.0, y2 + grow_y)
        mask = Image.new("L", (width, height), 0)
        inner = Image.new("L", (max(1, int((x2 - x1) * width)), max(1, int((y2 - y1) * height))), 255)
        mask.paste(inner, (int(x1 * width), int(y1 * height)))
        feather = max(2, int(min(width, height) * feather_fraction))
        return mask.filter(ImageFilter.GaussianBlur(feather))

    @staticmethod
    def to_latent_mask(mask: Image.Image, latent_height: int, latent_width: int) -> np.ndarray:
        # Block-average the (H, W) mask to the latent grid so a partially covered
        # 16x16 patch blends proportionally instead of flipping hard.
        array = np.asarray(mask.resize((latent_width * 16, latent_height * 16), Image.BILINEAR), dtype=np.float32)
        array = array.reshape(latent_height, 16, latent_width, 16).mean(axis=(1, 3)) / 255.0
        return array  # (latent_height, latent_width) in [0, 1]
