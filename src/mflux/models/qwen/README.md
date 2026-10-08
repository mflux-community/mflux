# Qwen Image
This directory contains MFLUX’s MLX implementation of the **Qwen Image** family, including both **text-to-image generation** and **natural language image editing**.

## Qwen Image (text-to-image)
**Qwen Image** is a 20B parameter text-to-image model ([technical report](https://arxiv.org/abs/2508.02324)). It uses a vision-language architecture with a 7B text encoder (Qwen2.5-VL) to understand and generate images from natural language descriptions.

The Qwen Image model has its own dedicated command `mflux-generate-qwen`. Qwen Image excels at multilingual prompts, including Chinese characters, and can render Chinese text as part of the image content (like signs, menus, and calligraphy).

![Qwen Image Examples](../../assets/qwen_image_example.jpg)

### Example: Wildlife Portrait

```sh
mflux-generate-qwen \
  --prompt "Close-up portrait of a majestic tiger in its natural habitat, detailed fur texture, piercing eyes, natural forest background, soft natural lighting, wildlife photography, photorealistic, high detail, professional wildlife shot" \
  --negative-prompt "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, text, letters, cartoon, anime, painting, drawing, illustration, 3d render, cgi, zoo, cage, artificial" \
  --width 1920 \
  --height 816 \
  --steps 30 \
  --seed 42 \
  -q 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

model = QwenImage(
    quantize=8,
    model_config=ModelConfig.qwen_image(),
)
image = model.generate_image(
    seed=42,
    prompt="Close-up portrait of a majestic tiger in its natural habitat, detailed fur texture, piercing eyes, natural forest background, soft natural lighting, wildlife photography, photorealistic, high detail, professional wildlife shot",
    negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, text, letters, cartoon, anime, painting, drawing, illustration, 3d render, cgi, zoo, cage, artificial",
    num_inference_steps=30,
    width=1920,
    height=816,
)
image.save("qwen_tiger.png")
```

You can also call the steps of `mflux-generate-qwen` and `mflux-generate-qwen-edit` from Python. `QwenImageCommand` (in `mflux.models.qwen.cli.qwen_image_generate`) and `QwenImageEditCommand` (in `mflux.models.qwen.cli.qwen_image_edit_generate`) each have `validate(args)`, `load(args)` and `generate(model, args, seed, prompt)`. They work like the Z-Image Turbo steps; the [Z-Image README](../z_image/README.md#z-image-turbo-example) has a full script and the rules for keeping a model loaded. Things to know for Qwen:

- A process that keeps the model loaded also keeps the text encoder: about 14 GB, or 15.5 GB for edits, which also load its vision part. `--quantize` does not shrink it, because mflux leaves the text encoder in bfloat16 on purpose (about 7 billion parameters at 2 bytes each). The exception is a pre-quantized checkpoint that ships its own quantized text encoder.
- `--model` takes this command's own names, a repo id or a local path. Another model's built-in name, such as `dev`, raises `ModelConfigError` before any weight is loaded. With a repo id or path, `--base-model` must also name this command's model, by alias or repo id.
- When `--guidance` is not given, `generate()` uses 3.5 for text-to-image and 2.5 for edits.
- `QwenImage` keeps each prompt's text embeddings in `model.prompt_cache`, one entry per distinct pair of prompt and negative prompt. A process that stays up and sees many different prompts should clear it now and then with `model.prompt_cache.clear()`, then `gc.collect()` and `mx.clear_cache()`. `QwenImageEdit` does not use that cache.
</details>

<details>
<summary><strong>Click to expand additional example commands</strong></summary>

**Chinese Calligraphy:**

```sh
mflux-generate-qwen \
  --prompt "Traditional Chinese calligraphy studio, ancient scrolls with beautiful Chinese characters, ink brushes, inkstone, traditional paper, warm natural lighting, peaceful atmosphere, photorealistic, high detail, cultural heritage" \
  --negative-prompt "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, text, letters, cartoon, anime, painting, drawing, illustration, 3d render, cgi, modern, digital" \
  --width 1920 \
  --height 816 \
  --steps 30 \
  --seed 42 \
  -q 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

model = QwenImage(
    quantize=8,
    model_config=ModelConfig.qwen_image(),
)
image = model.generate_image(
    seed=42,
    prompt="Traditional Chinese calligraphy studio, ancient scrolls with beautiful Chinese characters, ink brushes, inkstone, traditional paper, warm natural lighting, peaceful atmosphere, photorealistic, high detail, cultural heritage",
    negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, text, letters, cartoon, anime, painting, drawing, illustration, 3d render, cgi, modern, digital",
    num_inference_steps=30,
    width=1920,
    height=816,
)
image.save("qwen_calligraphy.png")
```
</details>

**Chinese Street Signs:**

```sh
mflux-generate-qwen \
  --prompt "Traditional Chinese street scene, old neighborhood with shop signs displaying Chinese characters (店铺, 餐厅, 书店), red lanterns, narrow alleys, traditional architecture, bustling street life, natural lighting, photorealistic, high detail, street photography" \
  --negative-prompt "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, cartoon, anime, painting, drawing, illustration, 3d render, cgi, modern signs, English text only" \
  --width 1920 \
  --height 816 \
  --steps 30 \
  --seed 42 \
  -q 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

model = QwenImage(
    quantize=8,
    model_config=ModelConfig.qwen_image(),
)
image = model.generate_image(
    seed=42,
    prompt="Traditional Chinese street scene, old neighborhood with shop signs displaying Chinese characters (店铺, 餐厅, 书店), red lanterns, narrow alleys, traditional architecture, bustling street life, natural lighting, photorealistic, high detail, street photography",
    negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, cartoon, anime, painting, drawing, illustration, 3d render, cgi, modern signs, English text only",
    num_inference_steps=30,
    width=1920,
    height=816,
)
image.save("qwen_street.png")
```
</details>

**Food Photography:**

```sh
mflux-generate-qwen \
  --prompt "Professional food photography, gourmet Chinese cuisine, steamed dumplings, colorful vegetables, traditional table setting, restaurant lighting, shallow depth of field, photorealistic, high detail, magazine quality" \
  --negative-prompt "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, text, letters, cartoon, anime, painting, drawing, illustration, 3d render, cgi, fast food, unappetizing" \
  --width 1920 \
  --height 816 \
  --steps 30 \
  --seed 42 \
  -q 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

model = QwenImage(
    quantize=8,
    model_config=ModelConfig.qwen_image(),
)
image = model.generate_image(
    seed=42,
    prompt="Professional food photography, gourmet Chinese cuisine, steamed dumplings, colorful vegetables, traditional table setting, restaurant lighting, shallow depth of field, photorealistic, high detail, magazine quality",
    negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, duplicate, watermark, signature, text, letters, cartoon, anime, painting, drawing, illustration, 3d render, cgi, fast food, unappetizing",
    num_inference_steps=30,
    width=1920,
    height=816,
)
image.save("qwen_food.png")
```
</details>

</details>

> [!WARNING]
> Note: The Qwen Image model requires downloading the `Qwen/Qwen-Image-2512` model weights (~58GB for the full model, or use quantization for smaller sizes).

## Qwen Image Edit (natural language image editing)
**Qwen Image Edit** enables precise natural language image editing, allowing you to modify images using text instructions while maintaining their original structure and context. The model uses a vision-language encoder to understand both the input image and your editing instructions.

Qwen Image Edit supports natural language editing with descriptive text instructions, maintains original poses and body positions when requested, supports multiple images for complex compositions, and works seamlessly with LoRA adapters for specialized transformations like camera angles and styles. The command runs `Qwen/Qwen-Image-Edit-2509` by default. `--model qwen-image-edit-2511` (Python: `ModelConfig.qwen_image_edit_2511()`) runs `Qwen/Qwen-Image-Edit-2511`, which reads its reference images at timestep 0 (`zero_cond_t` in diffusers).

![Qwen Image Edit Examples](../../assets/qwen_edit_example.jpg)
*Examples showing dog replacement with two-image input and monkey camera angle transformations with LoRAs. Source images: [Golden Retriever](https://images.unsplash.com/photo-1552053831-71594a27632d), [Grey Dog](https://images.unsplash.com/photo-1566710582818-d673dc761201), and [Monkey](https://images.unsplash.com/photo-1578948610588-ffe24448f5ed).*

### Example 1: Two-Image Transformation (Dog Replacement)

```sh
mflux-generate-qwen-edit \
  --image-paths "dog1.png" "dog2.png" \
  --prompt "Replace the golden retriever (standing outside, holding white rose) in Image 1 with the grey dog from Image 2 (which is standing inside in a studio). The grey dog should hold a red rose in its mouth and stand outside in the same position as the golden retriever. Maintain the outside environment, background, lighting, and all surroundings completely unchanged." \
  --steps 30 \
  --guidance 2.5 \
  --width 624 \
  --height 1024
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

model = QwenImageEdit(model_config=ModelConfig.qwen_image_edit())
image = model.generate_image(
    seed=42,
    prompt="Replace the golden retriever (standing outside, holding white rose) in Image 1 with the grey dog from Image 2 (which is standing inside in a studio). The grey dog should hold a red rose in its mouth and stand outside in the same position as the golden retriever. Maintain the outside environment, background, lighting, and all surroundings completely unchanged.",
    image_paths=["dog1.png", "dog2.png"],
    num_inference_steps=30,
    guidance=2.5,
    width=624,
    height=1024,
)
image.save("qwen_edit_dogs.png")
```

To run the command itself step by step with its own flags, use `QwenImageEditCommand` from `mflux.models.qwen.cli.qwen_image_edit_generate`. It has the same `validate(args)`, `load(args)` and `generate(model, args, seed, prompt)` steps as `QwenImageCommand` in the text-to-image section, with a default guidance of 2.5. Its `generate()` raises `ValueError` when `args.image_paths` is empty or `None`. The command line always sets it, so this only matters when you build `args` yourself.
</details>

### Example 2: Single Image with LoRAs (Camera Angle Transformations)

```sh
mflux-generate-qwen-edit \
  --image-paths "monkey.png" \
  --prompt "将镜头极度拉近，使用超长焦镜头进行极端特写拍摄，主体占据画面的大部分空间，背景完全虚化，营造出强烈的视觉冲击力和亲密感。Extreme zoom in with a super telephoto lens, creating an intense close-up where the subject dominates most of the frame, with the background completely blurred, creating a strong visual impact and sense of intimacy." \
  --steps 8 \
  --guidance 2.5 \
  --width 1024 \
  --height 1024 \
  --lora-paths "lightx2v/Qwen-Image-Lightning" "dx8152/Qwen-Edit-2509-Multiple-angles" \
  --lora-scales 0.5 1.0
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

model = QwenImageEdit(
    model_config=ModelConfig.qwen_image_edit(),
    lora_paths=["lightx2v/Qwen-Image-Lightning", "dx8152/Qwen-Edit-2509-Multiple-angles"],
    lora_scales=[0.5, 1.0],
)
image = model.generate_image(
    seed=42,
    prompt="将镜头极度拉近，使用超长焦镜头进行极端特写拍摄，主体占据画面的大部分空间，背景完全虚化，营造出强烈的视觉冲击力和亲密感。Extreme zoom in with a super telephoto lens, creating an intense close-up where the subject dominates most of the frame, with the background completely blurred, creating a strong visual impact and sense of intimacy.",
    image_paths=["monkey.png"],
    num_inference_steps=8,
    guidance=2.5,
    width=1024,
    height=1024,
)
image.save("qwen_edit_monkey.png")
```
</details>

*Uses [Qwen Image Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Lightning) for fast generation and [Camera Angle LoRA](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles) for precise camera control.*

### Tips for Qwen Image Edit
1. **Detailed Prompts**: The model works best with detailed, specific editing instructions
2. **Pose Maintenance**: Explicitly mention maintaining poses, body positions, or overall stance when you want to preserve the original structure
3. **Single Focus**: Focus on one or a few related edits at a time for more predictable results
4. **LoRA Combinations**: Combine multiple LoRAs for complex effects (e.g., fast generation + camera control)
5. **Quantization**: 6-bit or below can degrade the image a lot more compared to Flux, use with caution
6. **Seed Variation**: Qwen models typically do not vary much with seed changes. If you want more variation, vary the prompt instead
7. **Image Quality**: Qwen images come out quite soft compared to Flux models
8. **Output Size**: By default, edits keep the first input image size. Use `--width`/`--height` or scale factors like `2x` when you want resizing.

> [!WARNING]
> Note: The Qwen Image Edit model requires downloading the `Qwen/Qwen-Image-Edit-2509` model weights, or `Qwen/Qwen-Image-Edit-2511` for 2511 (~58GB for the full model, or use quantization for smaller sizes).
