# FIBO 2
This directory contains MFLUX's MLX implementation of **FIBO 2**.

MFLUX supports [Fibo 2.0 Dev](https://huggingface.co/briaai/Fibo-2.0-dev) from [Bria](https://bria.ai) for text-to-image generation and image editing, in one model, and [Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) for fast editing. FIBO 2 is an 8.1B-parameter flow-matching transformer trained on long structured JSON captions, with a Qwen3-VL-4B text encoder and the FLUX.2 VAE. It is not a new version of the [FIBO](../fibo/README.md) architecture: the transformer, the text encoder and the VAE are all different.

Quantization (`-q 4`, `-q 6`, `-q 8`), `mflux-save`, LoRA, `--low-ram`, step-by-step previews and metadata export work as for other models. Image-to-image with `--image-path` and training are not supported; for changing an existing image, use the editing command.

![FIBO 2 Example](../../assets/fibo2_example.jpg)

*Fibo 2.0 Dev, 1024×1024, 30 steps, guidance 5: a full JSON caption, a short description and a caption with text, all from the examples below.*

> [!WARNING]
> The weights are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) for non-commercial use. Commercial use needs a separate agreement with Bria. The repository is gated: accept the terms on the [Hugging Face model page](https://huggingface.co/briaai/Fibo-2.0-dev) and log in with `hf auth login` before the first download.

## Example
FIBO 2 is trained on structured JSON captions. Pass the caption as the prompt:

```sh
mflux-generate-fibo2 \
  --prompt '{"short_description": "A red vintage bicycle leaning against a whitewashed wall on a narrow Mediterranean street.", "objects": [{"description": "A red vintage bicycle with a wicker basket of lemons on its handlebars.", "location": "center", "relative_size": "large within frame"}], "background_setting": "A whitewashed wall with a blue wooden door and potted red geraniums.", "lighting": {"conditions": "bright morning sun", "direction": "from the right", "shadows": "long and crisp"}, "aesthetics": {"composition": "centered", "color_scheme": "red, white and blue", "mood_atmosphere": "calm"}, "photographic_characteristics": {"depth_of_field": "deep", "focus": "sharp on the bicycle", "camera_angle": "eye level"}, "style_medium": "photograph", "context": "A travel photograph.", "artistic_style": "realistic"}' \
  --seed 42
```

`--steps` defaults to 30 and `--guidance` to 5.0. The model card suggests 50 steps for the best quality. Width and height must be multiples of 16. Long captions are easier to keep in a file with `--prompt-file caption.json`.

<details>
<summary>Python API</summary>

```python
import json

from mflux.models.fibo2.variants import Fibo2

caption = {
    "short_description": "A red vintage bicycle leaning against a whitewashed wall on a narrow Mediterranean street.",
    "objects": [
        {"description": "A red vintage bicycle with a wicker basket of lemons on its handlebars.", "location": "center"}
    ],
    "background_setting": "A whitewashed wall with a blue wooden door and potted red geraniums.",
    "style_medium": "photograph",
}
model = Fibo2()
image = model.generate_image(seed=42, prompt=json.dumps(caption))
image.save("bicycle.png")
```
</details>

## Prompts
FIBO 2 was trained on three kinds of prompt, and mflux passes each of them on the way training saw it:

- **A full JSON caption**, as above. mflux drops empty values and the `aesthetic_score` and `preference_score` fields, which no training caption had, and serializes the caption compactly.
- **A short description as plain text**:

  ```sh
  mflux-generate-fibo2 \
    --prompt "A snowy owl perched on a mossy branch in a misty pine forest at dawn, soft golden light." \
    --seed 42
  ```

- **Exactly a `short_description` and a `text_render` list**, for text in the image. The list is kept even when it is empty, which tells the model the image has no text:

  ```sh
  mflux-generate-fibo2 \
    --prompt '{"short_description": "A cozy bakery storefront at dawn with a hand-painted wooden sign above the door.", "text_render": [{"text": "FRESH BREAD"}]}' \
    --seed 42
  ```

Freeform text that is not one of these forms works less well. To turn a short request into a full caption, see the prompt-to-JSON models on the [Bria FIBO](https://huggingface.co/briaai/FIBO) page.

With guidance on, the default negative prompt is the null caption the model was trained with: the caption keys with every value empty. Pass `--negative-prompt` to use your own.

### Text in images
When a caption asks for text in its `text_render` entries, mflux also shows Qwen3-VL each text drawn on its own, the way training drew it: one line, black on white, in Pillow's built-in font, with the fonts that ship with the model (Aileron, then DejaVu Sans) for characters that font lacks. FIBO 2 renders text better with them. A generation runs its first steps, while the noise level is above 0.9, on the caption without the images; an edit uses them at every step. A short description gets no images. Turn them off with `--no-text-images`, or `render_text_images=False` in Python.

## Editing
`mflux-generate-fibo2-edit` edits one to five images. The prompt is the edit instruction:

```sh
mflux-generate-fibo2-edit \
  --image-paths bicycle.png \
  --prompt "Make the bicycle blue" \
  --seed 42
```

Without `--width` and `--height`, the result takes the size FIBO 2 was trained at for the aspect ratio of the first image, about one megapixel. A single image is cropped to the size of the result; several images are each cropped to the trained size for their own aspect ratio.

The prompt can also be a JSON object with an `"edit_instruction"` key and a caption of the result, either as the caption's own keys or as a `"structured_caption"` key holding a caption or a short description:

```sh
mflux-generate-fibo2-edit \
  --image-paths bicycle.png \
  --prompt '{"structured_caption": "A blue vintage bicycle leaning against a whitewashed wall.", "edit_instruction": "Make the bicycle blue"}' \
  --seed 42
```

**Several images**: refer to them in order as `<image_1>`, `<image_2>` and so on:

```sh
mflux-generate-fibo2-edit \
  --image-paths bicycle.png owl.png \
  --prompt "Put the snowy owl from <image_2> in the wicker basket of the bicycle in <image_1>" \
  --seed 42
```

**Masked edits**: `--mask-path` marks the region of a single image to regenerate, white on black, at the image's size. As in training, the region is greyed out before the image is encoded:

```sh
mflux-generate-fibo2-edit \
  --image-paths owl.png \
  --mask-path mask.png \
  --prompt "Add a large full moon glowing through the mist" \
  --seed 42
```

<details>
<summary>Python API</summary>

```python
from mflux.models.fibo2.variants import Fibo2Edit

model = Fibo2Edit()
image = model.generate_image(seed=42, prompt="Make the bicycle blue", image_paths=["bicycle.png"])
image.save("blue_bicycle.png")
```
</details>

## Edit Turbo
[Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) is distilled from Dev for editing in 4 steps without guidance, so each step is one transformer pass instead of two. It has the same architecture, text encoder and VAE as Dev; only the transformer weights differ. It is for editing only: `mflux-generate-fibo2` does not accept it.

```sh
mflux-generate-fibo2-edit \
  --model fibo2-edit-turbo \
  --image-paths bicycle.png \
  --prompt "Make the bicycle blue" \
  --seed 42
```

`--steps` defaults to 4 and guidance to 1.0, which turns it off; `--negative-prompt` then has no effect. A 1024×1024 edit takes about 40 seconds on an M5 Pro, against about 10 minutes with Dev. The first run downloads another 25.5 GB, since the Hugging Face cache keeps each repository on its own.

## LoRA
Both commands load FIBO 2 LoRAs with `--lora PATH [SCALE]`, for example ones trained with diffusers' [FIBO 2 DreamBooth script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_bria_fibo2.md), which adapts every linear layer of the transformer:

```sh
mflux-generate-fibo2 \
  --prompt "A snowy owl perched on a mossy branch in a misty pine forest at dawn, soft golden light." \
  --lora my_fibo2_lora.safetensors 0.8 \
  --seed 42
```

mflux reads the diffusers, PEFT and ComfyUI key names, and applies each LoRA at the strength it was trained with, including the `lora_alpha` diffusers keeps in the file's metadata.

## Notes
- **Implementation**: the transformer is the Z-Image block with gated cross-attention to the text encoder in five layers, and a Perceiver that turns the text encoder output into 256 to 512 "gist" tokens. mflux reuses the FLUX.2 VAE, the Z-Image feed-forward, timestep, RoPE and final layers, and the shared Qwen3-VL language model and vision tower (also used by Qwen-Image 2.1), with 3D positions and deepstack features for images. The port follows the [diffusers implementation](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/bria_fibo2) and matches its transformer and text encoder in float32.
- **Speed**: about 9 to 12 seconds per step at 1024×1024 on an M5 Pro, since guidance runs the transformer twice per step.

## Quantization
Quantize on load with `-q`, or save a quantized copy once with `mflux-save` and load it by passing its folder to `--model`:

```sh
mflux-save --model fibo2 -q 8 --path fibo2-q8
mflux-generate-fibo2 --model fibo2-q8 --prompt "A red fox sitting in fresh snow, morning light" --seed 7
```

| Precision | Size on disk | Peak memory (1024×1024) |
|---|---|---|
| bf16 | 25.5 GB | 30.0 GB |
| q8 | 13 GB | 22.4 GB |
| q6 | 9.6 GB | 20.4 GB |
| q4 | 6.7 GB | 18.3 GB |

q8 and q6 are close to bf16. q4 keeps the subject and changes small details of the background.

> [!NOTE]
> The first run downloads about 25.5 GB from Hugging Face: a 16.3 GB transformer, an 8.9 GB text encoder and a 0.3 GB VAE.
