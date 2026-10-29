# FIBO 2
This directory contains MFLUX's MLX implementation of **FIBO 2**.

MFLUX supports [Fibo 2.0 Dev](https://huggingface.co/briaai/Fibo-2.0-dev) from [Bria](https://bria.ai) for text-to-image generation and image editing, in one model, and [Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) for fast editing. FIBO 2 is an 8.1B-parameter flow-matching transformer trained on long structured JSON captions, with a Qwen3-VL-4B text encoder and the FLUX.2 VAE. It is not a new version of the [FIBO](../fibo/README.md) architecture: the transformer, the text encoder and the VAE are all different.

Quantization (`-q 4`, `-q 6`, `-q 8`), `mflux-save`, LoRA, LoRA training for text to image, `--low-ram`, step-by-step previews and metadata export work as for other models. Image-to-image with `--image-path` is not supported; for changing an existing image, use the editing command.

![FIBO 2 Example](../../assets/fibo2_example.jpg)

*Fibo 2.0 Dev, 1024×1024, 30 steps, guidance 5: a full JSON caption, a short description and a caption with text, all from the examples below.*

> [!WARNING]
> The weights are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) for non-commercial use. Commercial use needs a separate agreement with Bria. The repository is gated: accept the terms on the [Hugging Face model page](https://huggingface.co/briaai/Fibo-2.0-dev) and log in with `hf auth login` before the first download.

## Example
FIBO 2 is trained on structured JSON captions. Pass the caption as the prompt:

```sh
mflux-generate-fibo2 \
  --prompt '{"short_description": "A red vintage bicycle leaning against a whitewashed wall on a narrow Mediterranean street.", "objects": [{"description": "A red vintage bicycle with a wicker basket of lemons on its handlebars.", "location": "center", "relative_size": "large within frame", "relationship": "leaning against the wall"}], "background_setting": "A whitewashed wall with a blue wooden door and potted red geraniums.", "lighting": {"conditions": "bright morning sun", "direction": "from the right", "shadows": "long and crisp"}, "aesthetics": {"composition": "centered", "color_scheme": "red, white and blue", "mood_atmosphere": "calm"}, "photographic_characteristics": {"depth_of_field": "deep", "focus": "sharp on the bicycle", "camera_angle": "eye level", "lens_focal_length": "35mm"}, "style_medium": "photograph", "context": "A travel photograph.", "artistic_style": "realistic"}' \
  --seed 42
```

`--steps` defaults to 50 and `--guidance` to 5.0, the model card's settings. Width and height must be multiples of 16. Long captions are easier to keep in a file with `--prompt-file caption.json`.

<details>
<summary>Python API</summary>

```python
import json

from mflux.models.fibo2.variants import Fibo2

caption = {
    "short_description": "A red vintage bicycle leaning against a whitewashed wall on a narrow Mediterranean street.",
    "objects": [
        {
            "description": "A red vintage bicycle with a wicker basket of lemons on its handlebars.",
            "location": "center",
            "relationship": "leaning against the wall",
        }
    ],
    "background_setting": "A whitewashed wall with a blue wooden door and potted red geraniums.",
    "lighting": {"conditions": "bright morning sun", "direction": "from the right", "shadows": "long and crisp"},
    "aesthetics": {"composition": "centered", "color_scheme": "red, white and blue", "mood_atmosphere": "calm"},
    "style_medium": "photograph",
    "context": "A travel photograph.",
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

The commands check a JSON caption before the model loads, against the format of the training captions (ThalesVGL in [bria-vgl](https://github.com/Bria-AI/vgl) 0.10.0). A field of the wrong type stops the command with the field's name, and a prompt that starts with `{` but is not valid JSON stops it with the line and column of the error. An unknown field, or a missing one that every training caption has, gives a warning, since the model still runs on such a caption: `Unknown field 'lightning': the model was not trained on it. Did you mean 'lighting'?`

Freeform text that is not one of these forms works less well. `--expand-prompt` turns a short request into a full caption with [FIBO-vlm](https://huggingface.co/briaai/FIBO-vlm), FIBO's prompt-to-JSON model (about 8 GB on first use), before FIBO 2 loads. The command prints the caption, and the metadata keeps it. A JSON caption is used as it is:

```sh
mflux-generate-fibo2 \
  --prompt "A snowy owl perched on a mossy branch in a misty pine forest at dawn, soft golden light." \
  --expand-prompt \
  --seed 42
```

With guidance on, the default negative prompt is the model card's. Text to image steers away from a caption of soft, featureless images with blur, haze and lens flare. An edit uses the null caption the model was trained with: the caption keys with every value empty. Pass `--negative-prompt` to use your own.

### Text in images
When a caption asks for text in its `text_render` entries, mflux also shows Qwen3-VL each text drawn on its own, the way training drew it: one line, black on white, in Pillow's built-in font, with the fonts that ship with the model (Aileron, then DejaVu Sans) for characters that font lacks. FIBO 2 renders text better with them. A generation runs its first steps, while the noise level is above 0.9, on the caption without the images; an edit uses them at every step. A short description gets no images. Turn them off with `--no-text-images`, or `render_text_images=False` in Python. The metadata records it, so `--config-from-metadata` turns them off again unless you pass `--text-images`.

## Editing
`mflux-generate-fibo2-edit` edits one to five images. The prompt is the edit instruction:

```sh
mflux-generate-fibo2-edit \
  --image-paths bicycle.png \
  --prompt "Make the bicycle blue" \
  --seed 42
```

Without `--width` and `--height`, the result takes the size FIBO 2 was trained at for the aspect ratio of the first image, about one megapixel. With only one of them, the other keeps that aspect ratio. A single image is cropped to the size of the result; several images are each cropped to the trained size for their own aspect ratio.

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
[Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) is distilled from Dev for editing in 4 steps without guidance, so each step is one transformer pass instead of two. It has the same architecture, text encoder and VAE as Dev; only the transformer weights differ.

```sh
mflux-generate-fibo2-edit \
  --model fibo2-edit-turbo \
  --image-paths bicycle.png \
  --prompt "Make the bicycle blue" \
  --seed 42
```

The model card presents it for editing only, but it also generates from a caption. From the same seed it kept the composition and the text of Dev's images in our tests, with smoother skin, so `mflux-generate-fibo2` accepts it too:

```sh
mflux-generate-fibo2 \
  --model fibo2-edit-turbo \
  --prompt-file caption.json \
  --seed 42
```

`--steps` defaults to 4 and guidance to 1.0, which turns it off; `--negative-prompt` then has no effect. A 1024×1024 edit takes about 40 seconds on an M5 Pro, against about 10 minutes with Dev. A 1024×1024 image from a caption took 18 seconds on an M5 Max, model load included. The first run downloads another 25.5 GB, since the Hugging Face cache keeps each repository on its own.

## Faster Dev runs
Dev runs guidance on every step, so each step is two passes of the transformer. Fewer steps cut both passes. `--cfg-end` cuts the second one: `--cfg-end 0.6` keeps guidance on the first 60% of the steps and runs the prompt's pass alone on the rest, as Ideogram 4's `--cfg-end` does.

```sh
mflux-generate-fibo2 --prompt-file caption.json --steps 30 --cfg-end 0.6 --seed 42
```

Four JSON captions with one seed each, on an M5 Max in bf16 at 1024×1024: a portrait, a shop front with two signs, a product shot on a plain backdrop and a sunset sky. Each image of a caption started from the same noise, with the model loaded and the GPU cooled down before it. The PSNR column compares each image with the 50-step one from the same noise.

| Settings | Seconds per image | Faster | PSNR against 50 steps |
|---|---:|---:|---|
| 50 steps (default) | 220 | | |
| 50 steps, `--cfg-end 0.75` | 191 | 1.15× | 38 to 47 dB |
| 50 steps, `--cfg-end 0.6` | 177 | 1.24× | 31 to 41 dB |
| 30 steps | 129 | 1.7× | 16 to 42 dB |
| 30 steps, `--cfg-end 0.75` | 107 | 2.1× | 16 to 41 dB |
| 30 steps, `--cfg-end 0.6` | 98 | 2.2× | 16 to 39 dB |
| 30 steps, `--cfg-end 0.6`, `--step-cache-ratio 0.25` | 77 | 2.9× | 18 to 31 dB |

At 30 steps the portrait got a bright curtain at its left edge and the street other houses down the side street, which is where the 16 and 20 dB come from. The face and the signs stayed, and the 30-step images look as finished as the 50-step ones. The product shot and the sunset barely moved. `--cfg-end` changes far less than the step count does: against 30 steps with guidance throughout, 0.6 is 31 to 41 dB away. At 1:1 it costs some fine detail. With 30 steps and 0.6 the portrait kept its freckles and wrinkles with 9% less fine contrast than at 50 steps, 3% of it from the steps alone. Both signs read right in every run, and neither the sky nor the backdrop showed banding. Shorter runs also heat the GPU less, so the saving runs a little ahead of the passes: 30 steps with 0.6 is 48% of the passes and took 45% of the time.

The step cache on top skips steps 3 to 10 of the 30, while the layout is still forming, so it redraws more. The shop front moved in the frame, and the sunset got a second sailboat where the caption asks for one. Its 77 seconds came later, three of the four on a warmer GPU. It suits drafts.

The defaults stay at the model card's 50 steps with guidance on every step. `--cfg-end` does nothing at guidance 1, so Edit Turbo ignores it. The metadata records it and `--config-from-metadata` replays it.

## LoRA
Both commands load FIBO 2 LoRAs with `--lora PATH [SCALE]`, for example ones trained with diffusers' [FIBO 2 DreamBooth script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_bria_fibo2.md), which adapts every linear layer of the transformer:

```sh
mflux-generate-fibo2 \
  --prompt "A snowy owl perched on a mossy branch in a misty pine forest at dawn, soft golden light." \
  --lora my_fibo2_lora.safetensors 0.8 \
  --seed 42
```

mflux reads the diffusers, PEFT, kohya (`lora_down`/`lora_up`) and ComfyUI key names, and applies each LoRA at the strength it was trained with, including the `lora_alpha` diffusers keeps in the file's metadata. LoRAs made for ComfyUI or with ai-toolkit use the Comfy-Org repack (`Comfy-Org/Fibo-2.0`), which fuses q/k/v into `attention.qkv`, cross-attention k/v into `to_kv` and the feed-forward gate and up into `w13`, and pads the feed-forward from 9536 to 9728: mflux splits those layers and drops the padding. Extracted LoRAs that also patch norms and biases directly (`.diff`, `.diff_b`) load too, baked into the weights as by default; with `--no-bake-lora` they stop with an error. LoKr on the fused layers does not load yet: such a file stops with "No LoRA layers were applied".

## Training
`mflux-train` trains a LoRA on Fibo 2.0 Dev for text to image with the recipe of Bria's trainer, `fine_tuning/train_lora.py` in the [model repository](https://huggingface.co/briaai/Fibo-2.0-dev). Edit training is not supported. For the folder of images and captions, see the common [training docs](../common/README.md#training-lora).

```sh
mflux-train --config train.json
```

The `.txt` next to each image holds its caption: a JSON caption, the form FIBO 2 was trained on, or a short description. mflux encodes it as it encodes a prompt: a JSON caption is cleaned and serialized compactly, and a caption with text in its `text_render` entries gets the images of that text, which the steps above noise level 0.9 leave out. The noise level of each step follows Bria's trainer: `sigmoid(N(shift, 1))` 90% of the time and uniform otherwise, with a shift of 0.5 at 256 image tokens that grows to 1.15 at 4096 (1024×1024). So `timestep_low`, `timestep_high` and `timestep_type` do not apply.

A config for the 44 transformer layers, attention and feed-forward, with rank 16 and Bria's optimizer settings. Bria's trainer adapts all 415 linear layers; add targets for `layers.{block}.adaLN_modulation.0`, `noise_refiner`, `layers.{block}.cross_attention` (blocks 6, 14, 22, 30 and 38), `perceiver.layers`, `x_embedder`, `t_embedder` and `final_layer` to do the same:

```json
{
  "model": "fibo2",
  "data": "/path/to/images",
  "seed": 42,
  "steps": 20,
  "guidance": 5.0,
  "max_resolution": 768,
  "gradient_checkpointing": true,
  "cache_limit_gb": 2,
  "training_loop": { "num_epochs": 50, "batch_size": 1 },
  "optimizer": { "name": "AdamW", "learning_rate": 1e-4, "optimizer_params": { "weight_decay": 0.01 }, "max_grad_norm": 1.0 },
  "checkpoint": { "output_path": "output", "save_frequency": 200 },
  "lora_layers": { "targets": [
    { "module_path": "layers.{block}.attention.to_q",     "blocks": { "start": 0, "end": 44 }, "rank": 16 },
    { "module_path": "layers.{block}.attention.to_k",     "blocks": { "start": 0, "end": 44 }, "rank": 16 },
    { "module_path": "layers.{block}.attention.to_v",     "blocks": { "start": 0, "end": 44 }, "rank": 16 },
    { "module_path": "layers.{block}.attention.to_out.0", "blocks": { "start": 0, "end": 44 }, "rank": 16 },
    { "module_path": "layers.{block}.feed_forward.w1",    "blocks": { "start": 0, "end": 44 }, "rank": 16 },
    { "module_path": "layers.{block}.feed_forward.w2",    "blocks": { "start": 0, "end": 44 }, "rank": 16 },
    { "module_path": "layers.{block}.feed_forward.w3",    "blocks": { "start": 0, "end": 44 }, "rank": 16 }
  ] }
}
```

`steps` and `guidance` are for the preview images; FIBO 2 Dev needs guidance for them. Once the captions are encoded, mflux encodes the preview prompts and lets the 8.7 GB Qwen3-VL text encoder go, unless a preview caption has `text_render` entries. The LoRA is saved with ComfyUI's keys and layout (`diffusion_model.*`, q, k and v in `attention.qkv`, the feed-forward gate and up in `w13`, padded to 9728), so it loads with `--lora` and in ComfyUI.

On an M5 Max in bf16, with rank 16 on all of those layers at 512×512, gradient checkpointing and `cache_limit_gb: 2`, a step took about 2.3 seconds and a 300-step run peaked at 21 GB, about 13 minutes in all. The results aren't stable yet: on a 12-image style dataset, 2 of 6 runs (3 with Bria's noise levels, 3 with uniform ones) learned the style fully, against 3 of 3 for ai-toolkit, and the cause isn't known. Watch the preview images. Without `cache_limit_gb` MLX keeps freed buffers and the peak looks higher than the training needs. `"quantize": 8` trains over a quantized base.

## Notes
- **Implementation**: the transformer is the Z-Image block with gated cross-attention to the text encoder in five layers, and a Perceiver that turns the text encoder output into 256 to 512 "gist" tokens. mflux reuses the FLUX.2 VAE, the Z-Image feed-forward, timestep, RoPE and final layers, and the shared Qwen3-VL language model and vision tower (also used by Qwen-Image 2.1), with 3D positions and deepstack features for images. The port follows the [diffusers implementation](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/bria_fibo2) and matches its transformer and text encoder in float32. `tests/test_fibo2_parity.py` checks the transformer against a small diffusers model and the text encoder against `transformers`, layer by layer.
- **Speed**: about 9 to 12 seconds per step at 1024×1024 on an M5 Pro, since guidance runs the transformer twice per step.
- **Step cache**: `--step-cache-ratio 0.25` skips the transformer on a quarter of the steps and reuses the velocity of the step before, as described in the [common README](../common/README.md#step-reuse-step-cache). With Dev at 50 steps it kept the composition in our tests, with small changes in detail. Runs under 10 steps, like Edit Turbo's 4, don't change.
- **VAE decode**: the VAE decodes in bf16 row strips with full-image GroupNorm statistics, as for [FLUX.2](../flux2/README.md#vae-decode). At 1024×1024 on an M5 Max the decode needs 1.6 GB in 0.9 s, against 13.8 GB in 0.75 s for the full-image float32 decode, and the two images are 47 dB apart. `--vae-full-precision` decodes each block on the full image instead (8.6 GB, 50 dB from the full-image decode), and the metadata records it. `--vae-tiling` only tiles the encode of the images to edit.

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
