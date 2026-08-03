import json
import textwrap
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.common_models.qwen3_vl.qwen3_vl_util import Qwen3VLUtil
from mflux.models.fibo_vlm.fibo_vlm_initializer import FiboVLMInitializer
from mflux.models.fibo_vlm.tokenizer.qwen2vl_processor import Qwen2VLProcessor
from mflux.utils.gpu_stream import with_gpu_stream

EDIT_SYSTEM_PROMPT = '''You are a highly skilled Creative Director for Visual Adaptation at a leading Generative AI company.

Your expertise lies in using an existing image as a visual reference to create entirely new scenes. You can deconstruct a reference image to understand its subject, pose, and style, and then reimagine it in a new context based on textual instructions.

Your primary task is to receive a reference image and a textual edit instruction. You will analyze the reference to extract key visual information and then generate a comprehensive JSON object describing a new scene that creatively incorporates the user's instructions.

**FIDELITY RULE:** Unless explicitly requested or implied by the user's edit instruction (e.g., changing the setting requires new lighting), you should preserve all visual properties of the reference image. This includes, but is not limited to: the identity, pose, and appearance of all subjects (people/animals), the existence, location, size, and orientation of all objects, the overall composition, camera angle, lens characteristics, and the style/medium. Only change what is strictly required to fulfill the user's edit.
E.g. if the user's request is to change the expression of a person in the image, the only change should be the expression, not the pose, composition, background, etc. On the other hand, if the user requested to change the background to a snowy forest, you should change the background, lighting, and mood atmosphere, etc, but keep the subject the same.
Make sure to acknowledge the changing elements in the relevant "description" fields (i.e. "short_description", object "description", etc.). e.g. if the user requested to change the expression of a person to "sad", make sure to write "the person is sad" in the "short_description".

Adhere strictly to the following structure and guidelines:

1.  **Inputs:** You will receive a reference image and a textual edit instruction. You will NOT receive a starting JSON.
2.  **Core Logic (Analyze and Synthesize):**
    * **Analyze:** First, deeply analyze the provided reference image. Identify its primary subject(s), their specific poses, expressions, and appearance. Also note the overall composition, lighting style, and artistic medium.
    * **Synthesize:** Next, interpret the textual edit instruction to understand what elements to keep from the reference and what to change. You will then construct a brand new JSON object from scratch that describes the desired final scene. For example, if the instruction is "the same dog and pose, but at the beach," you must describe the dog from the reference image in the `objects` array but create a new `background_setting` for a beach, with appropriate `lighting` and `mood_atmosphere`. **The description of every unchanged element (objects, people, composition, camera settings, style) MUST be derived directly from the reference image.**
3.  **Output:** Your output MUST be ONLY a single, valid JSON object that describes the **new, imagined scene**. Do not describe the original reference image.
4.  **Holistic Consistency:** Ensure the generated JSON is internally consistent. A change in the environment should be reflected logically across multiple fields, such as `background_setting`, `lighting`, `shadows`, and the `short_description`.
5.  **Schema Adherence:** The new JSON object you generate must strictly follow the schema provided below.

Unless the user's prompt explicitly requests a different style (e.g., 'painting', 'cartoon', 'illustration') and if the original image is a photograph, you MUST default to `style_medium: "photograph"`. In such cases, `artistic_style` should be "realistic" or a similar descriptor.

**Edit Instruction Rewriting Rule:** You should rewrite the user's request into a single instruction in the imperative style. Follow the examples below to guide your rewriting:

| Edit Category | User Input Example | Your Rewritten Output (Imperative Style) |
| :--- | :--- | :--- |
| **Modify Style** | "I want this picture to look like a cartoon." | `Turn the image into the cartoon style.` |
| | "Could you make it an oil painting?" | `Change the image style to Oil_Painting.` |
| **Object Attribute** | "Make the dog black and white instead of brown."| `Change the dog's color to black and white.` |
| | "I want the shirt to be a formal white shirt."| `Change the subject's shirt to a formal white shirt.` |
| **Adding Elements** | "Put a hat on the person's head." | `Add a wide-brimmed felt hat to the subject.` |
| **Removing Objects** | "Get rid of the book they are holding." | `Remove the book from the subject's hands.` |
| **Replacing Objects** | "Replace the rose with a sunflower." | `Change the rose to a bright yellow sunflower.` |
| **Lighting/Relight** | "Change the lighting to be bright and vibrant." | `Change the lighting from dark and moody to bright and vibrant.` |
| **Composition/Camera** | "I want a wider shot, showing more of the background."| `Change the perspective to a wider shot.` |
| **Text/Label** | "Change the text 'Happy Anniversary' to 'Hello'."| `Change the text "Happy Anniversary" to "Hello".` |
| **Quality/Refinement** | "Can you make the image sharper?" | `Refine the image to obtain increased clarity and sharpness.` |
| **Style Reference** | "Use the style in the image but with a fox." | `Use the style in the image to depict a red fox walking.` |

The JSON object must contain the following keys precisely:

1.  `short_description`: (String) A concise summary of the imagined image content, 200 words maximum.
2.  `objects`: (Array of Objects) List a maximum of 5 prominent objects. If the scene implies more than 5, creatively choose the most important ones and describe the rest in the background. For each object, include:
    * `description`: (String) A detailed description of the imagined object, 100 words maximum.
    * `location`: (String) E.g., "center", "top-left", "bottom-right foreground".
    * `relative_size`: (String) E.g., "small", "medium", "large within frame". (If a person is the main subject, this should be "medium-to-large" or "large within frame").
    * `shape_and_color`: (String) Describe the basic shape and dominant color.
    * `texture`: (String) E.g., "smooth", "rough", "metallic", "furry".
    * `appearance_details`: (String) Any other notable visual details.
    * `relationship`: (String) Describe the relationship between the object and the other objects in the image.
    * `orientation`: (String) Describe the orientation or positioning of the object, e.g., "upright", "tilted 45 degrees", "horizontal", "vertical", "facing left", "facing right", "upside down", "lying on its side".
    * If the object is a human or a human-like object, include the following:
        * `pose`: (String) Describe the body position.
        * `expression`: (String) Describe facial expression and emotion. E.g., "winking", "joyful", "serious", "surprised", "calm".
        * `clothing`: (String) Describe attire.
        * `action`: (String) Describe the action of the human.
        * `gender`: (String) Describe the gender of the human.
        * `skin_tone_and_texture`: (String) Describe the skin tone and texture.
    * If the object is a cluster of objects, include the following:
        * `number_of_objects`: (Integer) The number of objects in the cluster.
3.  `background_setting`: (String) Describe the overall environment, setting, or background, including any notable background elements that are not part of the `objects` section.
4.  `lighting`: (Object)
    * `conditions`: (String) E.g., "bright daylight", "dim indoor", "studio lighting", "golden hour".
    * `direction`: (String) E.g., "front-lit", "backlit", "side-lit from left".
    * `shadows`: (String) Describe the presence and quality of shadows, e.g., "long, soft shadows", "sharp, defined shadows", "minimal shadows".
5.  `aesthetics`: (Object)
    * `composition`: (String) E.g., "rule of thirds", "symmetrical", "centered", "leading lines". If people are the main subject, specify the shot type, e.g., "medium shot", "close-up", "portrait composition".
    * `color_scheme`: (String) E.g., "monochromatic blue", "warm complementary colors", "high contrast".
    * `mood_atmosphere`: (String) E.g., "serene", "energetic", "mysterious", "joyful".
6.  `photographic_characteristics`: (Object)
    * `depth_of_field`: (String) E.g., "shallow", "deep", "bokeh background".
    * `focus`: (String) E.g., "sharp focus on subject", "soft focus", "motion blur".
    * `camera_angle`: (String) E.g., "eye-level", "low angle", "high angle", "dutch angle".
    * `lens_focal_length`: (String) E.g., "wide-angle", "telephoto", "macro", "fisheye". (If the main subject is a person, prefer "standard lens (e.g., 35mm-50mm)" or "portrait lens (e.g., 50mm-85mm)" to ensure they are framed more closely. Avoid "wide-angle" for people unless specified).
7.  `style_medium`: (String) Identify the artistic style or medium based on the user's prompt or creative interpretation (e.g., "photograph", "oil painting", "watercolor", "3D render", "digital illustration", "pencil sketch").
8.  `artistic_style`: (String) If the style is not "photograph", describe its specific artistic characteristics, 3 words maximum. (e.g., "impressionistic, vibrant, textured" for an oil painting).
9.  `context`: (String) Provide a general description of the type of image this would be. For example: "This is a concept for a high-fashion editorial photograph intended for a magazine spread," or "This describes a piece of concept art for a fantasy video game."
10. `text_render`: (Array of Objects) By default, this array should be empty (`[]`). Only add text objects to this array if the user's prompt explicitly specifies the exact text content to be rendered (e.g., user asks for "a poster with the title 'Cosmic Dream'"). Do not invent titles, names, or slogans for concepts like book covers or posters unless the user provides them. A rare exception is for universally recognized text that is integral to an object (e.g., the word 'STOP' on a 'stop sign'). For all other cases, if the user does not provide text, this array must be empty.
    * `text`: (String) The exact text content provided by the user. NEVER use generic placeholders.
    * `location`: (String) E.g., "center", "top-left", "bottom-right foreground".
    * `size`: (String) E.g., "medium", "large", "large within frame".
    * `color`: (String) E.g., "red", "blue", "green".
    * `font`: (String) E.g., "realistic", "cartoonish", "minimalist", "serif typeface".
    * `appearance_details`: (String) Any other notable visual details.
11. `edit_instruction`: (String) **The generated string must be phrased as a single, direct, and imperative command to a text-to-image model (e.g., "Change the subject's outfit to a spacesuit", "Turn the image into a watercolor painting style. Add a cat to the image"), starting with a strong action verb like: Add, Change, Make, Move, Remove, Render, Turn, Zoom in/out, etc. ** It must clearly and concisely describe all the intended changes introduced by the user's edit and **MUST NOT** explicitly refer to a "reference image" or "original image."'''

EDIT_USER_PROMPT_TEMPLATE = """[EDIT INSTRUCTIONS]:
{user_prompt}

[TASK]:
Use the provided image as a visual reference only. Analyze its key elements (like the subject and pose) and then generate a new, detailed JSON object for the scene described in the instructions above. Do not describe the reference image itself; describe the new scene. Follow all of your system rules."""

EDIT_WITH_MASK_SYSTEM_PROMPT = """You are a highly skilled Creative Director for Visual Adaptation at a leading Generative AI company.

Your expertise lies in "Image Completion": using a reference image that contains grey masked areas and filling those areas based on textual instructions. You can deconstruct the visible parts of a reference image to understand its subject, style, and composition, and then seamlessly perform Inpainting, Outpainting, or Background Generation to complete the scene.

Your primary task is to receive a reference image (containing grey masks) and a textual edit instruction. You will generate a comprehensive JSON object describing the final, completed scene where the masked areas have been filled.

**FIDELITY RULE:** You must preserve all visual properties of the visible (non-masked) parts of the reference image. This includes the identity of subjects, existing objects, composition, and artistic style. Your task is to fill the grey masked regions so they blend perfectly with the unmasked areas. Only change what is strictly required to fulfill the user's edit within the masked zones.

Adhere strictly to the following structure and guidelines:

1.  **Inputs:** You will receive a reference image (with grey masks) and a textual edit instruction.
2.  **Core Logic (Analyze and Synthesize):**
    * **Analyze:** Identify which areas are masked. Determine the existing style, lighting, and subject matter from the visible portions.
    * **Synthesize:** Interpret the instruction to perform one of three tasks: **Inpainting/Generative-fill** (adding/restoring objects), **Outpainting/Image expansion** (extending the scene), or **Background Generation/Completion**.
3.  **Output:** Your output MUST be ONLY a single, valid JSON object describing the **completed image**. Do not describe the grey masks; describe the content that now occupies those spaces.
4.  **Edit Instruction Rewriting Rule:** You should rewrite the user's request into a single instruction in the imperative style, referring to the masked regions as "the masked area" or "the masked regions". Describe the expected content of the masked regions in detail, in the context of the entire image. If the user did not specify the required content for the masked regions, you should come up with the most plausible content that would complete the scene. Follow the examples below to guide your rewriting:

| Category / Intent | Your Rewritten Output (Imperative Style) |
| :--- | :--- |
| **Object Generation** | `Generate a white rose with a blue center in the masked region.` |
| **Detailed Extension** | `extend the image into the masked region to create a scene featuring A symmetrical, abstract composition featuring illuminated blue arrows and rectangles against a dark background. The arrows point towards the center, creating a sense of depth and perspective, while the rectangles add texture and complexity to the overall design.` |
| **Specific Addition** | `Put a chef hat with a yellow top and a red band in the masked region` |
| **Background Detail** | `Create the following background in the masked region: A vast, calm body of water, likely an ocean or large sea, extends to the horizon under a hazy sky.` |
| **Atmospheric Fill** | `Fill the background masked area with a clear, bright blue sky with very subtle, wispy white clouds, suggesting an open and expansive outdoor environment.` |
| **Environment Infill** | `Create inside the masked area: a greenhouse with rows of plants in pots, under a glass ceiling.` |
| **Subject Restoration** | `Restore the area in the mask with a young woman` |
| **Description Restoration** | `Restore the masked regions using the image description: A top-down view of a set of pink crocheted baby clothing items, including a hat, booties, and mittens, arranged vertically on a light brown wooden surface.` |

The JSON object must contain the following keys precisely:

**Items 1–10:** Identical to the Edit system prompt — `short_description`, `objects`, `background_setting`, `lighting`, `aesthetics`, `photographic_characteristics`, `style_medium`, `artistic_style`, `context`, `text_render`.

**Item 11 (edit_instruction, mask-specific):** `edit_instruction`: (String) **The generated string must be phrased as a single, direct, and imperative command to a text-to-image model, specifically directing it to fill the grey areas of the image. Use phrases like: "Complete the background in the masked regions...", "Restore the masked regions...", "Expand the canvas through the masked regions...", etc. ** It must clearly and concisely describe the content that should occupy the masked areas and refer to the "masked regions" or "masked area" as the target for the generation."""

EDIT_WITH_MASK_USER_PROMPT_TEMPLATE = """[EDIT INSTRUCTIONS]:
{user_prompt}

[TASK]:
Use the provided image as a visual reference, identifying the grey areas as regions to be filled or extended. Analyze the key elements in the visible parts (like the subject and pose) and then generate a new, detailed JSON object for the completed scene that follows the instructions above. Follow all of your system rules."""


class FiboVLM:
    decoder = Qwen3VLDecoder

    def __init__(
        self,
        model_id: str = "briaai/FIBO-vlm",
        model_path: str | None = None,
        quantize: int | None = None,
    ):
        FiboVLMInitializer.init(
            model=self,
            model_path=model_path if model_path else model_id,
            quantize=quantize,
        )

    @property
    def processor(self) -> Qwen2VLProcessor:
        return self.tokenizers["fibo_vlm"].processor

    @with_gpu_stream
    def generate(
        self,
        prompt: str,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: List[str] | None = None,
        seed: int | None = None,
    ) -> str:
        return self._generate_internal(
            task="generate",
            prompt=prompt,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        )

    def refine(
        self,
        structured_prompt: str,
        editing_instructions: str,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: List[str] | None = None,
        seed: int | None = None,
    ) -> str:
        return self._generate_internal(
            task="refine",
            structured_prompt=structured_prompt,
            editing_instructions=editing_instructions,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        )

    def inspire(
        self,
        image: Image.Image,
        prompt: str | None = None,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: List[str] | None = None,
        seed: int | None = None,
    ) -> str:
        return self._generate_internal(
            task="inspire",
            image=image,
            prompt=prompt,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        )

    def edit(
        self,
        image: Image.Image,
        edit_instruction: str,
        use_mask: bool = False,
        seed: int | None = None,
    ) -> str:
        try:
            return self._generate_internal(
                task="edit",
                image=image,
                editing_instructions=edit_instruction,
                top_p=0.9,
                temperature=0.2,
                max_tokens=4096,
                stop=["<|im_end|>", "<|end_of_text|>"],
                seed=seed,
                use_mask=use_mask,
            )
        except ValueError as exc:
            if use_mask:
                raise ValueError(
                    "Masked edit prompt generation failed; refusing fallback that could ignore the masked regions."
                ) from exc

            # The explicit edit prompt is preferable, but the local FIBO VLM can
            # occasionally emit non-JSON responses for it. Fall back to the
            # image-conditioned local refine format so the edit CLI still works.
            fallback_response = self._generate_internal(
                task="refine",
                refine_image=image,
                editing_instructions=edit_instruction,
                top_p=0.9,
                temperature=0.2,
                max_tokens=4096,
                stop=["<|im_end|>", "<|end_of_text|>"],
                seed=seed,
            )
            try:
                return FiboVLM._normalize_edit_output(
                    fallback_response,
                    fallback_edit_instruction=edit_instruction,
                )
            except ValueError:
                raise exc

    def _generate_internal(
        self,
        task: str,
        *,
        prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        refine_image: Optional[Image.Image] = None,
        structured_prompt: Optional[str] = None,
        editing_instructions: Optional[str] = None,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop: List[str] | None = None,
        seed: int | None = None,
        use_mask: bool = False,
    ) -> str:
        stop = stop or ["<|im_end|>", "<|end_of_text|>"]
        messages = FiboVLM._build_messages(
            task=task,
            image=image,
            prompt=prompt,
            refine_image=refine_image,
            structured_prompt=FiboVLM._normalize_json(structured_prompt),
            editing_instructions=editing_instructions,
            use_mask=use_mask,
        )
        formatted = FiboVLM._process_messages(self.processor, messages, image, refine_image)
        input_ids = mx.array(formatted["input_ids"])
        attention_mask = mx.array(formatted["attention_mask"]) if formatted.get("attention_mask") is not None else None
        pixel_values = mx.array(formatted["pixel_values"]) if formatted.get("pixel_values") is not None else None
        image_grid_thw = mx.array(formatted["image_grid_thw"]) if formatted.get("image_grid_thw") is not None else None
        stop_token_sequences = [self.processor.tokenizer.encode(s, add_special_tokens=False) for s in stop]
        generated_ids = Qwen3VLUtil.generate_text(
            seed=seed,
            top_p=top_p,
            decoder=self.decoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop_token_sequences=stop_token_sequences,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_tokens = generated_ids[:, input_ids.shape[1] :]
        generated_text = self.processor.tokenizer.decode(np.array(generated_tokens[0]), skip_special_tokens=True)
        if task == "edit":
            return FiboVLM._normalize_edit_output(generated_text)
        return FiboVLM._format_json_output(generated_text)

    @staticmethod
    def _normalize_json(structured_prompt: Optional[str]) -> Optional[str]:
        if structured_prompt is None:
            return None
        try:
            return json.dumps(json.loads(structured_prompt.strip()), separators=(",", ":"), ensure_ascii=False)
        except json.JSONDecodeError:
            return structured_prompt

    @staticmethod
    def _process_messages(
        processor: Qwen2VLProcessor,
        messages: List[Dict[str, Any]],
        image: Optional[Image.Image],
        refine_image: Optional[Image.Image],
    ) -> Dict[str, Any]:
        has_images = image is not None or refine_image is not None

        if has_images:
            images_list = FiboVLM._extract_images(messages)
            prompt_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = {"text": [prompt_text], "padding": True, "return_tensors": "np"}
            if images_list:
                inputs["images"] = images_list
            return processor(**inputs)

        return processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="np", return_dict=True
        )

    @staticmethod
    def _extract_images(messages: List[Dict[str, Any]]) -> List[Image.Image]:
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        img = item.get("image")
                        if img is not None:
                            images.append(img)
        return images

    @staticmethod
    def _format_json_output(text: str) -> str:
        try:
            return f"\n{json.dumps(json.loads(text), indent=2, ensure_ascii=False)}\n"
        except json.JSONDecodeError:
            return text

    @staticmethod
    def _build_messages(
        task: str,
        *,
        image: Optional[Image.Image] = None,
        refine_image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        structured_prompt: Optional[str] = None,
        editing_instructions: Optional[str] = None,
        use_mask: bool = False,
    ) -> List[Dict[str, Any]]:
        if task == "inspire":
            inspire_text = "<inspire>"
            if prompt:
                inspire_text = f"<inspire>\n{(prompt or '').strip()}"
            content = [{"type": "image", "image": image}, {"type": "text", "text": inspire_text}]
        elif task == "generate":
            content = [{"type": "text", "text": f"<generate>\n{(prompt or '').strip()}"}]
        elif task == "edit":
            return FiboVLM._build_edit_messages(
                image=image, editing_instructions=editing_instructions, use_mask=use_mask
            )
        else:
            content = FiboVLM._build_refine_content(refine_image, structured_prompt, editing_instructions)

        return [{"role": "user", "content": content}]

    @staticmethod
    def _build_edit_messages(
        image: Optional[Image.Image],
        editing_instructions: Optional[str],
        use_mask: bool,
    ) -> List[Dict[str, Any]]:
        if image is None:
            raise ValueError("Edit mode requires an image.")

        user_prompt = (editing_instructions or "").strip()
        system_prompt = EDIT_WITH_MASK_SYSTEM_PROMPT if use_mask else EDIT_SYSTEM_PROMPT
        user_text_template = EDIT_WITH_MASK_USER_PROMPT_TEMPLATE if use_mask else EDIT_USER_PROMPT_TEMPLATE
        user_text = user_text_template.format(user_prompt=user_prompt)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}],
            },
        ]

    @staticmethod
    def _build_refine_content(
        refine_image: Optional[Image.Image],
        structured_prompt: Optional[str],
        editing_instructions: Optional[str],
    ) -> List[Dict[str, Any]]:
        edits = (editing_instructions or "").strip()

        if refine_image is None:
            base_prompt = (structured_prompt or "").strip()
            text = textwrap.dedent(
                f"""<refine>
Input:
{base_prompt}
Editing instructions:
{edits}"""
            ).strip()
            return [{"type": "text", "text": text}]

        text = textwrap.dedent(
            f"""<refine>
Editing instructions:
{edits}"""
        ).strip()
        return [{"type": "image", "image": refine_image}, {"type": "text", "text": text}]

    @staticmethod
    def _normalize_edit_output(text: str, fallback_edit_instruction: str | None = None) -> str:
        payload = FiboVLM._extract_single_json_object(text)
        aesthetics = payload.setdefault("aesthetics", {})
        if not isinstance(aesthetics, dict):
            raise ValueError("VLM edit output must contain an object at `aesthetics`.")
        aesthetics.setdefault("preference_score", "very high")
        aesthetics.setdefault("aesthetic_score", "very high")

        payload = FiboVLM._remove_empty_values(payload)
        edit_instruction = payload.get("edit_instruction")
        if not isinstance(edit_instruction, str) or not edit_instruction.strip():
            normalized_fallback = FiboVLM._normalize_fallback_edit_instruction(fallback_edit_instruction)
            if normalized_fallback is not None:
                payload["edit_instruction"] = normalized_fallback
            else:
                raise ValueError("VLM edit output must include a non-empty `edit_instruction`.")

        edit_instruction = payload.get("edit_instruction")
        if not isinstance(edit_instruction, str) or not edit_instruction.strip():
            raise ValueError("VLM edit output must include a non-empty `edit_instruction`.")

        return f"\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n"

    @staticmethod
    def _extract_single_json_object(text: str) -> Dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                candidate = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

        objects: List[Dict[str, Any]] = []
        start_index: int | None = None
        depth = 0
        in_string = False
        escaped = False

        for index, char in enumerate(candidate):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                if depth == 0:
                    start_index = index
                depth += 1
                continue

            if char == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start_index is not None:
                    snippet = candidate[start_index : index + 1]
                    try:
                        parsed = json.loads(snippet)
                    except json.JSONDecodeError:
                        start_index = None
                        continue
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    start_index = None

        if not objects:
            raise ValueError("VLM edit output must contain exactly one JSON object.")
        return max(objects, key=FiboVLM._score_edit_json_candidate)

    @staticmethod
    def _score_edit_json_candidate(candidate: Dict[str, Any]) -> tuple[int, int]:
        score = 0
        if isinstance(candidate.get("edit_instruction"), str) and candidate["edit_instruction"].strip():
            score += 10
        if isinstance(candidate.get("short_description"), str) and candidate["short_description"].strip():
            score += 3
        if isinstance(candidate.get("objects"), list):
            score += 2
        if isinstance(candidate.get("lighting"), dict):
            score += 2
        if isinstance(candidate.get("aesthetics"), dict):
            score += 3

        return score, len(json.dumps(candidate, ensure_ascii=False))

    @staticmethod
    def _remove_empty_values(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {
                key: FiboVLM._remove_empty_values(item) for key, item in value.items() if item not in (None, "", {}, [])
            }
            return {key: item for key, item in cleaned.items() if item not in (None, "", {}, [])}
        if isinstance(value, list):
            cleaned_list = [FiboVLM._remove_empty_values(item) for item in value]
            return [item for item in cleaned_list if item not in (None, "", {}, [])]
        return value

    @staticmethod
    def _normalize_fallback_edit_instruction(edit_instruction: str | None) -> str | None:
        if edit_instruction is None:
            return None

        normalized = edit_instruction.strip()
        if not normalized:
            return None

        return normalized[0].upper() + normalized[1:]
