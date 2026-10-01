# FIBO
[FIBO](https://huggingface.co/briaai/FIBO) from [Bria.ai](https://bria.ai), the first open-source JSON-native text-to-image model trained on long structured captions. FIBO delivers high image quality, strong prompt adherence, and professional-grade control—trained exclusively on licensed data. ([Technical Paper](https://arxiv.org/abs/2511.06876))

![FIBO Example](../../assets/fibo_example.jpg)

FIBO is an 8B-parameter DiT-based, flow-matching model using **SmolLM3-3B** as the text encoder with a novel **DimFusion** conditioning architecture for efficient long-caption training, and **Wan 2.2** as the VAE. The VLM-assisted prompting uses a fine-tuned **Qwen3-VL** to expand short user intents, fill in missing details, and extract/edit structured prompts from images.

Most text-to-image models excel at imagination—but not control. FIBO is trained on structured JSON captions up to 1,000+ words, enabling precise, reproducible control over lighting, composition, color, and camera settings. The structured captions foster native disentanglement, allowing targeted, iterative refinement without prompt drift.

## Key Features
- **VLM-guided JSON-native prompting**: Transform short prompts into structured schemas with 1,000+ words (lighting, camera, composition, DoF)
- **Disentangled control**: Tweak a single attribute (e.g., camera angle) without breaking the scene
- **Strong prompt adherence**: High alignment on PRISM-style evaluations
- **Enterprise-grade**: 100% licensed data with governance, repeatability, and legal clarity

## FIBO Lite

[FIBO Lite](https://huggingface.co/briaai/Fibo-lite) is a two-stage distilled variant combining CFG distillation and SCFM for fast few-step generation. Use `--model fibo-lite` for ~10x speed: 8 steps, `guidance=1.0`, no negative prompt needed. Slight quality tradeoff vs. base FIBO.

```sh
mflux-generate-fibo \
  --model fibo-lite \
  --prompt "A tiny watercolor robot in a garden" \
  --steps 8 \
  --seed 42
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

vlm = FiboVLM()
json_prompt = vlm.generate(prompt="A tiny watercolor robot in a garden", seed=42)
model = FIBO(model_config=ModelConfig.fibo_lite())
image = model.generate_image(
    seed=42,
    prompt=json_prompt,
    num_inference_steps=8,
)
image.save("robot_lite.png")
```
</details>

## The four modes: Generate, Edit, Refine, and Inspire

### Generate
While the actual prompt input to FIBO is a structured JSON file, the generate command provides an interface to input pure text prompts. These are then expanded into structured JSON prompts using FIBO's Vision-Language Model (VLM) before being passed to the diffusion model for image generation.

```sh
mflux-generate-fibo \
  --prompt "Three cartoon animal chefs in a colorful bakery kitchen, Pixar style: a bunny with floppy ears wearing a tall white chef hat and pink apron holding a chocolate cake on the left, a raccoon with a striped tail wearing blue oven mitts and a yellow bandana frosting cupcakes in the center, a penguin wearing a red bowtie and checkered apron carrying a tray of golden croissants on the right, warm kitchen lighting with flour dust in air" \
  --width 1200 \
  --height 540 \
  --steps 20 \
  --guidance 4.0 \
  --seed 42 \
  --output animal_bakers.png
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.models.fibo_vlm.model.util import FiboVLMUtil

vlm = FiboVLM()
json_prompt = vlm.generate(
    prompt="Three cartoon animal chefs in a colorful bakery kitchen, Pixar style: a bunny with floppy ears wearing a tall white chef hat and pink apron holding a chocolate cake on the left, a raccoon with a striped tail wearing blue oven mitts and a yellow bandana frosting cupcakes in the center, a penguin wearing a red bowtie and checkered apron carrying a tray of golden croissants on the right, warm kitchen lighting with flour dust in air",
    seed=42,
)
FiboVLMUtil.save_json_prompt(Path("animal_bakers.json"), json_prompt)

model = FIBO(model_config=ModelConfig.fibo())
image = model.generate_image(
    seed=42,
    prompt=json_prompt,
    num_inference_steps=50,
    width=1200,
    height=540,
    guidance=4.0,
)
image.save("animal_bakers.png")
```
</details>

This command will output both the generated image (`animal_bakers.png`) and a JSON prompt file (`animal_bakers.json`) containing the expanded structured prompt used for generation.

If a JSON prompt file is provided, it will be used directly for image generation, thus bypassing the VLM step and giving you full control over the prompt structure:

```sh
mflux-generate-fibo \
    --prompt-file animal_bakers.json \
    --width 1200 \
    --height 540 \
    --steps 50 \
    --guidance 4.0 \
    --seed 42 \
    --output animal_bakers.png
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo_vlm.model.util import FiboVLMUtil

structured_prompt = FiboVLMUtil.get_structured_prompt(Path("animal_bakers.json"))
model = FIBO(model_config=ModelConfig.fibo())
image = model.generate_image(
    seed=42,
    prompt=structured_prompt,
    num_inference_steps=50,
    width=1200,
    height=540,
    guidance=4.0,
)
image.save("animal_bakers.png")
```
</details>

When working with a JSON prompt file, you can use whatever tool you prefer to edit it and are not forced to use the built-in FIBO-VLM. Other good alternatives are [coding](https://cursor.com/agents) [agents](https://www.claude.com/product/claude-code), other [LLMs](https://github.com/ml-explore/mlx-lm)/[VLMs](https://github.com/Blaizzy/mlx-vlm) etc.

<details>
<summary><strong>Click to expand the JSON prompt file used (animal_bakers.json)</strong></summary>

```json
{
  "short_description": "Three cartoon animal chefs are in a bakery kitchen, each holding a culinary creation. A bunny chef on the left presents a chocolate cake, a raccoon chef in the center is frosting cupcakes, and a penguin chef on the right carries a tray of croissants. The kitchen is brightly lit with warm tones, and flour dusts the air, creating a lively and cheerful baking atmosphere.",
  "objects": [
    {
      "description": "A cartoon bunny wearing a white chef's hat and a pink apron, holding a chocolate cake with white frosting and cherries.",
      "location": "left foreground",
      "relationship": "The bunny chef is presenting the chocolate cake.",
      "relative_size": "medium",
      "shape_and_color": "Rounded bunny shape, white hat, pink apron, brown cake, white frosting, red cherries.",
      "texture": "smooth",
      "appearance_details": "Floppy ears, rosy cheeks, smiling expression.",
      "pose": "Standing upright, holding the cake with both hands.",
      "expression": "Joyful and proud.",
      "clothing": "White chef's hat, pink apron.",
      "action": "Holding and presenting a cake.",
      "gender": "female",
      "skin_tone_and_texture": "White fur, smooth texture.",
      "orientation": "Upright, facing forward."
    },
    {
      "description": "A cartoon raccoon wearing blue oven mitts and a yellow bandana, actively frosting cupcakes with pink and yellow frosting.",
      "location": "center midground",
      "relationship": "The raccoon chef is in the process of frosting cupcakes.",
      "relative_size": "medium",
      "shape_and_color": "Distinct raccoon shape, blue mitts, yellow bandana, pink and yellow frosting, brown cupcakes.",
      "texture": "smooth",
      "appearance_details": "Striped tail, bushy fur, focused expression.",
      "pose": "Leaning forward slightly, hands busy frosting.",
      "expression": "Concentrated and happy.",
      "clothing": "Blue oven mitts, yellow bandana.",
      "action": "Frosting cupcakes.",
      "gender": "male",
      "skin_tone_and_texture": "Brown and black fur, smooth texture.",
      "orientation": "Upright, slightly angled."
    },
    {
      "description": "A cartoon penguin wearing a red bowtie and a checkered apron, carrying a tray of golden croissants.",
      "location": "right foreground",
      "relationship": "The penguin chef is carrying a tray of freshly baked croissants.",
      "relative_size": "medium",
      "shape_and_color": "Classic penguin shape, red bowtie, red and white apron, golden croissants.",
      "texture": "smooth",
      "appearance_details": "Black and white body, orange beak and feet, smiling expression.",
      "pose": "Standing upright, holding the tray with both hands.",
      "expression": "Cheerful and friendly.",
      "clothing": "Red bowtie, checkered apron.",
      "action": "Carrying a tray of croissants.",
      "gender": "male",
      "skin_tone_and_texture": "Feathered texture, smooth appearance.",
      "orientation": "Upright, facing forward."
    }
  ],
  "background_setting": "A brightly lit bakery kitchen with wooden countertops, shelves stocked with baking ingredients and equipment, and a window in the background. Flour is lightly dusted in the air.",
  "lighting": {
    "conditions": "warm indoor lighting",
    "direction": "front-lit and side-lit",
    "shadows": "soft, diffused shadows"
  },
  "aesthetics": {
    "composition": "centered composition with the three chefs forming a horizontal line",
    "color_scheme": "warm and cheerful, with dominant browns, yellows, pinks, and whites",
    "mood_atmosphere": "joyful, friendly, and inviting",
    "aesthetic_score": "very high",
    "preference_score": "very high"
  },
  "photographic_characteristics": {
    "depth_of_field": "shallow",
    "focus": "sharp focus on the chefs and their creations",
    "camera_angle": "eye-level",
    "lens_focal_length": "standard lens"
  },
  "style_medium": "digital illustration",
  "text_render": [],
  "context": "This image is a charming illustration, likely for a children's book, educational material, or a bakery-themed advertisement, designed to evoke feelings of happiness and the joy of baking.",
  "artistic_style": "Pixar-style animation"
}
```

Note: This JSON prompt was generated using FIBO's "Generate" mode from a short text description. Note the strong alignment between the JSON prompt and the image!
</details>

### Refine
While the JSON prompt can be edited manually, it can be quite complex and inconvenient to modify directly. The refinement mode expands a simple user instruction in order to tweak specific attributes. The VLM processes these instructions and updates the JSON prompt accordingly before generating new images.

![FIBO Refine Example](../../assets/fibo_refine_example.jpg)

Assuming we already have a previous prompt file, like `owl_brown.json`, we can refine this prompt to change the owl's color and add some accessories:

```sh
mflux-refine-fibo \
    --prompt-file owl_brown.json \
    --instructions "Make the owl white instead of brown, and add round glasses and a black scarf. Keep everything else exactly the same - the same forest background, moonlight lighting, composition, and overall whimsical atmosphere." \
    --output owl_white.json
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.models.fibo_vlm.model.util import FiboVLMUtil

vlm = FiboVLM()
refined_json = vlm.refine(
    seed=42,
    structured_prompt=FiboVLMUtil.get_structured_prompt(Path("owl_brown.json")),
    editing_instructions="Make the owl white instead of brown, and add round glasses and a black scarf. Keep everything else exactly the same - the same forest background, moonlight lighting, composition, and overall whimsical atmosphere.",
)
FiboVLMUtil.save_json_prompt(Path("owl_white.json"), refined_json)
```
</details>

Then generate the refined image using the updated JSON prompt:

```sh
mflux-generate-fibo \
    --prompt-file owl_white.json \
    --width 1024 \
    --height 560 \
    --steps 20 \
    --guidance 4.0 \
    --seed 42 \
    --quantize 4 \
    --output owl_white.png
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo_vlm.model.util import FiboVLMUtil

structured_prompt = FiboVLMUtil.get_structured_prompt(Path("owl_white.json"))
model = FIBO(model_config=ModelConfig.fibo(), quantize=4)
image = model.generate_image(
    seed=42,
    prompt=structured_prompt,
    num_inference_steps=50,
    width=1024,
    height=560,
    guidance=4.0,
)
image.save("owl_white.png")
```
</details>

It is worth noting that refine does not work the same way as other editing techniques like Flux Kontext or Qwen Image Edit. Instead of modifying an existing image, it modifies the underlying **structured prompt** to produce a new image.

### Inspire
Provide an image instead of text. FIBO's vision-language model extracts a detailed, structured prompt, blends it with your creative intent, and produces related images—ideal for inspiration without overreliance on the original.

![FIBO Inspire Example](../../assets/fibo_inspire_example.jpg)

Starting from an image, you can extract a structured JSON prompt that captures its visual characteristics:

```sh
mflux-inspire-fibo \
    --image-path bird.jpg \
    --prompt "blue and brown bird on brown tree trunk" \
    --output bird_inspired.json \
    --seed 42
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.models.fibo_vlm.model.util import FiboVLMUtil

vlm = FiboVLM()
image = FiboVLMUtil.load_image(Path("bird.jpg"))
inspired_json = vlm.inspire(
    seed=42,
    image=image,
    prompt="blue and brown bird on brown tree trunk",
)
FiboVLMUtil.save_json_prompt(Path("bird_inspired.json"), inspired_json)
```
</details>

Then generate new images with similar characteristics:

```sh
mflux-generate-fibo \
    --prompt-file bird_inspired.json \
    --width 1024 \
    --height 672 \
    --steps 50 \
    --guidance 4.0 \
    --seed 42 \
    -q 8 \
    --output bird_inspired.png
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo_vlm.model.util import FiboVLMUtil

structured_prompt = FiboVLMUtil.get_structured_prompt(Path("bird_inspired.json"))
model = FIBO(model_config=ModelConfig.fibo(), quantize=8)
image = model.generate_image(
    seed=42,
    prompt=structured_prompt,
    num_inference_steps=50,
    width=1024,
    height=672,
    guidance=4.0,
)
image.save("bird_inspired.png")
```
</details>

---

### FIBO-Edit
FIBO Edit supports direct image-conditioned editing using a plain-text edit instruction plus a source image.
The CLI will convert the instruction + image into a structured JSON prompt automatically.

![FIBO Edit Example](../../assets/fibo_edit_example.jpg)

```sh
mflux-generate-fibo-edit \
    --image-path reference_upscaled.png \
    --prompt "Make the hand fistbump the camera instead of showing a flat palm." \
    --width 640 \
    --height 384 \
    --steps 30 \
    --guidance 3.5 \
    --seed 42 \
    --output fibo_edit_fistbump.png
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.edit import FIBOEdit
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.models.fibo_vlm.model.util import FiboVLMUtil

vlm = FiboVLM(quantize=8)
structured_prompt = vlm.edit(
    image=FiboVLMUtil.load_image(Path("image.png")),
    edit_instruction="Make the hand fistbump the camera instead of showing a flat palm.",
    seed=42,
)

model = FIBOEdit(model_config=ModelConfig.fibo_edit(), quantize=8)
image = model.generate_image(
    seed=42,
    prompt=structured_prompt,
    image_path="image.png",
    num_inference_steps=30,
    width=640,
    height=384,
    guidance=3.5,
)
image.save("fibo_edit_fistbump.png")
```
</details>

Optional localized editing is supported with a mask:

![FIBO Edit Mask Example](../../assets/fibo_edit_mask_example.jpg)

```sh
mflux-generate-fibo-edit \
    --image-path reference_upscaled.png \
    --mask-path hand_mask.png \
    --prompt "Make only the masked hand fistbump the camera and keep the rest of the image unchanged." \
    --steps 30 \
    --guidance 3.5 \
    --seed 42 \
    --output fibo_edit_fistbump_masked.png
```

Advanced use: you can still pass a full structured JSON prompt via `--prompt` or `--prompt-file`.
When doing so, make sure the JSON includes `edit_instruction`.

<details>
<summary>Example full edit JSON</summary>

```json
{
  "short_description": "A close-up shot of a Black man's hand making a fistbump gesture towards the camera. He is wearing a plain white t-shirt. The background is a softly blurred indoor setting with a window and curtains.",
  "objects": [
    {
      "description": "A Black man's hand, with visible knuckles and skin texture, making a fistbump gesture.",
      "location": "center foreground",
      "relationship": "The hand is the primary subject, making contact with the camera.",
      "relative_size": "large within frame",
      "shape_and_color": "Human hand shape, dark brown skin tone.",
      "texture": "Smooth skin with visible knuckles.",
      "appearance_details": "Fingers are curled into a fist, thumb is extended.",
      "orientation": "facing forward, fist extended towards the viewer"
    },
    {
      "description": "A Black man's torso and lower face, partially visible, wearing a white t-shirt.",
      "location": "center midground",
      "relationship": "The man is the owner of the hand, providing context for the gesture.",
      "relative_size": "medium",
      "shape_and_color": "Human torso and face shape, dark brown skin tone, white shirt.",
      "texture": "Smooth skin, soft fabric of the t-shirt.",
      "appearance_details": "He has a short beard and mustache. His expression is serious and direct.",
      "pose": "Upper body visible, arm extended forward for a fistbump.",
      "expression": "serious, direct gaze",
      "clothing": "plain white crew-neck t-shirt",
      "action": "fistbumping the camera",
      "gender": "male",
      "skin_tone_and_texture": "dark brown, smooth skin",
      "orientation": "upright, facing forward"
    }
  ],
  "background_setting": "A softly blurred indoor setting, featuring a light gray wall on the left, a window with natural light streaming through on the right, and sheer white curtains partially drawn.",
  "lighting": {
    "conditions": "bright indoor lighting, natural light from a window",
    "direction": "side-lit from right",
    "shadows": "soft shadows are cast on the left side of the hand and face, indicating light from the right window."
  },
  "aesthetics": {
    "composition": "centered, portrait composition with the hand as the focal point",
    "color_scheme": "neutral tones with a pop of white from the shirt and natural light.",
    "mood_atmosphere": "direct, engaging, slightly serious.",
    "photographic_characteristics": {
      "depth_of_field": "shallow",
      "focus": "sharp focus on the hand and face, with a blurred background",
      "camera_angle": "eye-level",
      "lens_focal_length": "standard lens (e.g., 35mm-50mm)"
    },
    "style_medium": "photograph",
    "artistic_style": "realistic, naturalistic",
    "preference_score": "very high",
    "aesthetic_score": "very high"
  },
  "context": "This is a portrait photograph, potentially for a social media profile, a casual greeting, or a promotional image emphasizing a direct and engaging interaction.",
  "edit_instruction": "Make the hand fistbump the camera instead of showing a flat palm."
}
```
</details>


### FIBO-Edit-RMBG
Background removal via `briaai/Fibo-Edit-RMBG` (`--model fibo-edit-rmbg`). **`--output`** is a **transparent PNG** (cutout). You can omit **`--prompt`**—mflux uses a built-in matte instruction. Default **steps / guidance** for this model are **10 / 1.0** if you don’t set them.

![FIBO Edit RMBG Side-by-Side Example](../../assets/fibo_edit_remove_background_example.png)
*Original source image credit: [Unsplash](https://images.unsplash.com/photo-1587971051803-70bf6d4ae977?q=80&w=2274&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)*

Smallest useful command (writes **`image.png`** in the current directory):

```sh
mflux-generate-fibo-edit \
    --model fibo-edit-rmbg \
    --image-path tools_input_new_small.png
```

<details>
<summary>More detail (full CLI example)</summary>

RGBA cutout, grayscale matte, and a **`.metadata.json`** next to the cutout; explicit steps / guidance / seed.

```sh
mflux-generate-fibo-edit \
    --model fibo-edit-rmbg \
    --image-path tools_input_new_small.png \
    --output fibo_edit_rmbg_cutout.png \
    --matte-output fibo_edit_rmbg_matte.png \
    --make-conf \
    --steps 10 \
    --guidance 1.0 \
    --seed 42
```
</details>


## Notes
> [!WARNING]
> FIBO and FIBO-Edit require downloading the `briaai/FIBO`, `briaai/FIBO-lite`, `briaai/Fibo-Edit`, or `briaai/Fibo-Edit-RMBG` model weights (~24GB each), plus the `briaai/FIBO-vlm` vision-language model (~8GB), totaling ~32GB for a full setup with one FIBO-family model, or use quantization for smaller sizes.

