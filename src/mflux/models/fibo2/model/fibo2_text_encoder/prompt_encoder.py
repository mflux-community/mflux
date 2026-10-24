import json
import math

import mlx.core as mx
from PIL import Image

from mflux.models.common_models.qwen3_vl.qwen3_vl_processor import Qwen3VLProcessor
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_images import Fibo2TextImages


class Fibo2PromptEncoder:
    MAX_LENGTH = 4096
    PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    # Qwen3-VL layers behind the transformer's six text inputs: the Perceiver's, then one per injection block
    OUTPUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
    # The null caption FIBO 2 was trained with, the default negative prompt of an edit
    NULL_CAPTION = (
        '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
        '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
        '"context":"","artistic_style":""}'
    )
    # The model card's negative prompt for text to image (NEGATIVE_T2I): soft, featureless, artifact-laden images
    TEXT_TO_IMAGE_NEGATIVE = (
        '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
        'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, '
        'no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],"background_setting":'
        '"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture",'
        '"lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast","overexposed highlights",'
        '"lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":{"depth_of_field":"shallow, '
        'heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, lacking fine texture"},'
        '"style_medium":"","context":"","artistic_style":""}'
    )
    SCORE_KEYS = ("aesthetic_score", "preference_score")
    VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

    @staticmethod
    def encode_prompt(
        prompt: str,
        processor: Qwen3VLProcessor,
        text_encoder: Fibo2TextEncoder,
        images: list[Image.Image] | None = None,
    ) -> mx.array:
        text = Fibo2PromptEncoder.PROMPT_TEMPLATE.format(prompt)
        inputs = processor([text], images=images, return_tensors="np", truncation=True, max_length=Fibo2PromptEncoder.MAX_LENGTH)  # fmt: off
        pixel_values = mx.array(inputs["pixel_values"]) if images else None
        grids = mx.array(inputs["image_grid_thw"]) if images else None
        hidden_states = text_encoder(mx.array(inputs["input_ids"]), pixel_values, grids)

        # Each layer is scaled to unit RMS per token in float32, then every bundle concatenates its three layers
        layers = sorted({layer for bundle in Fibo2PromptEncoder.OUTPUT_LAYERS for layer in bundle})
        normalized = {layer: Fibo2PromptEncoder._unit_rms(hidden_states[layer]) for layer in layers}
        bundles = [mx.concatenate([normalized[layer] for layer in bundle], axis=-1) for bundle in Fibo2PromptEncoder.OUTPUT_LAYERS]  # fmt: off
        return mx.stack(bundles, axis=1)

    @staticmethod
    def edit_prompt(prompt: str, num_images: int, text_images: bool = False) -> str:
        # A vision marker per image, the result's caption if any, then the instruction
        caption, instruction = Fibo2PromptEncoder.edit_caption(prompt)
        if text_images and isinstance(caption, dict):
            Fibo2PromptEncoder._mark_text_entries(caption, Fibo2PromptEncoder.VISION_MARKER)
        if num_images == 1:
            edit = {"image": Fibo2PromptEncoder.VISION_MARKER}
        else:
            edit = {f"image_{k}": Fibo2PromptEncoder.VISION_MARKER for k in range(1, num_images + 1)}
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def format_prompt(prompt: str) -> str:
        # A JSON caption in its trained form, serialized compactly; a short description stays as it is
        caption, _ = Fibo2PromptEncoder._parse_edit_prompt(prompt)
        if isinstance(caption, str):
            return prompt
        return json.dumps(caption or {}, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def check_edit_prompt(prompt: str) -> None:
        # A JSON edit prompt names its instruction, and a "structured_caption" holds the whole caption
        try:
            edit = json.loads(prompt)
        except json.JSONDecodeError:
            return
        if not isinstance(edit, dict):
            return
        if "edit_instruction" not in edit:
            raise ValueError(f'A JSON edit prompt needs an "edit_instruction" key: {prompt}')
        caption = edit.get("structured_caption")
        if "structured_caption" in edit and (not isinstance(caption, (dict, str)) or len(edit) > 2):
            raise ValueError(f'A "structured_caption" must hold the whole caption, beside only "edit_instruction": {prompt}')  # fmt: off

    @staticmethod
    def clean_caption(caption: dict) -> dict:
        # Training captions hold no empty values and no aesthetic_score or preference_score
        keys = set(caption)
        cleaned = {
            key: {k: v for k, v in value.items() if k not in Fibo2PromptEncoder.SCORE_KEYS}
            if isinstance(value, dict)
            else value
            for key, value in Fibo2PromptEncoder._drop_empty(caption).items()
            if key not in Fibo2PromptEncoder.SCORE_KEYS
        }
        cleaned = {key: value for key, value in cleaned.items() if value != {}}
        # Exactly a short_description and a text_render list keeps both, in that order, and the list even when empty
        if keys == {"short_description", "text_render"}:
            cleaned = {
                "short_description": cleaned.get("short_description") or "",
                "text_render": cleaned.get("text_render") or [],
            }
        return cleaned

    @staticmethod
    def edit_caption(prompt: str) -> tuple[dict | str | None, str]:
        # A prompt without an "edit_instruction" is the instruction alone
        caption, instruction = Fibo2PromptEncoder._parse_edit_prompt(prompt)
        if instruction is None:
            return None, prompt
        return caption, instruction

    @staticmethod
    def mark_text_images(prompt: str, marker: str) -> str:
        # The marker is the vision marker of each text's image, or empty for the steps that run without them
        caption = Fibo2PromptEncoder._mark_text_entries(json.loads(prompt), marker)
        return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def caption(prompt: str) -> dict | None:
        # The caption of a formatted text-to-image prompt, or None for a short description
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError:
            return None
        return caption if isinstance(caption, dict) else None

    @staticmethod
    def _mark_text_entries(caption: dict, marker: str) -> dict:
        for entry in caption.get("text_render") or []:
            if Fibo2TextImages.has_text(entry):
                entry["image_tokens"] = marker
        return caption

    @staticmethod
    def _parse_edit_prompt(prompt: str) -> tuple[dict | str | None, str | None]:
        # The result's caption: the other keys, or "structured_caption" (a caption or a short description)
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError:
            return prompt, None
        if not isinstance(caption, dict):
            return prompt, None
        instruction = caption.pop("edit_instruction", None)
        if "structured_caption" in caption:
            caption = caption.pop("structured_caption")
            if not isinstance(caption, dict):
                return caption or None, instruction
        return Fibo2PromptEncoder.clean_caption(caption) or None, instruction

    @staticmethod
    def _drop_empty(value):
        if isinstance(value, dict):
            value = {k: Fibo2PromptEncoder._drop_empty(v) for k, v in value.items()}
            return {k: v for k, v in value.items() if not Fibo2PromptEncoder._is_empty(v)}
        if isinstance(value, list):
            return [v for v in map(Fibo2PromptEncoder._drop_empty, value) if not Fibo2PromptEncoder._is_empty(v)]
        return value

    @staticmethod
    def _is_empty(value) -> bool:
        if isinstance(value, float):
            return math.isnan(value)
        return value is None or (isinstance(value, (str, dict, list)) and not value)

    @staticmethod
    def _unit_rms(states: mx.array) -> mx.array:
        dtype = states.dtype
        states = states.astype(mx.float32)
        return (states * mx.rsqrt(mx.mean(states * states, axis=-1, keepdims=True) + 1e-6)).astype(dtype)
