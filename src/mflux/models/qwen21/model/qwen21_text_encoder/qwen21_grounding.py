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


class Qwen21Grounding:
    GROUNDING_PROMPT = GROUNDING_PROMPT
    REWRITE_PROMPT = REWRITE_PROMPT
    VERIFY_PROMPT = VERIFY_PROMPT

    @staticmethod
    def build_input_ids(tokenizer, n_image_tokens: int, query: str) -> list[int]:
        # Chat-shaped grounding request over one image; the caller expands each
        # <|image_pad|> placeholder to the image's merged token count before encoding.
        text = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            + GROUNDING_PROMPT.format(query=query)
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        return Qwen21Grounding.tokenize_with_images(tokenizer, text, [n_image_tokens])

    @staticmethod
    def tokenize_with_images(tokenizer, text: str, image_token_counts: list[int]) -> list[int]:
        # Expands the FIRST len(image_token_counts) <|image_pad|> placeholders in text
        # to each image's merged token count (one placeholder per image, in order),
        # then tokenizes the whole chat-shaped string.
        remaining = list(image_token_counts)
        parts = []
        for chunk in text.split("<|image_pad|>"):
            parts.append(chunk)
            if remaining:
                parts.append("<|image_pad|>" * remaining.pop(0))
        expanded = "".join(parts)
        ids = tokenizer.tokenizer(expanded, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):  # HF tokenizers nest the single sequence
            ids = ids[0]
        return list(ids)

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
        for candidate in reversed(Qwen21Grounding.balanced_json_spans(text)):
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
        for candidate in reversed(Qwen21Grounding.balanced_json_spans(text)):
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
            applied = '"instruction_applied": true' in lowered or "applied: true" in lowered
            unchanged = '"outside_unchanged": true' in lowered or "unchanged: true" in lowered
            return applied, unchanged
        return None

    @staticmethod
    def parse_bbox(text: str, image_size: tuple[int, int]) -> tuple[float, float, float, float] | None:
        # Returns (x1, y1, x2, y2) as fractions of image_size, or None when the reply
        # carries no plausible box. Three coordinate regimes exist: values <= 2 are
        # already fractions; values beyond the shown image's size follow the classic
        # Qwen-VL 0-1000 normalized convention; anything else is absolute pixels of
        # the shown image.
        width, height = image_size
        for match in _BBOX_PATTERN.finditer(text):
            x1, y1, x2, y2 = (float(v) for v in match.groups())
            if x2 <= x1 or y2 <= y1:
                continue
            if max(x1, y1, x2, y2) <= 2.0:
                box = (x1, y1, x2, y2)
            elif max(x1, y1, x2, y2) > max(width, height):
                box = (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
            else:
                box = (x1 / width, y1 / height, x2 / width, y2 / height)
            if max(box) > 1.5:
                continue
            clamped = tuple(min(max(v, 0.0), 1.0) for v in box)
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
