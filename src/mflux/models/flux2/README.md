# FLUX.2
This directory contains MFLUX's MLX implementation of **FLUX.2**.

MFLUX supports [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), and [FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) from Black Forest Labs, released in January 2026. FLUX.2 Klein is a fast, efficient image generation model available in 4B and 9B parameter variants. The 4B model delivers high-quality images in just 4 steps, making it one of the fastest open-source models available.

All the standard modes such as img2img, LoRA, LyCORIS LoKr, and quantizations are supported for this model. FLUX.2 also supports image-conditioned editing with multi-image support.

![FLUX.2 Klein Example](../../assets/flux2_klein.jpg)

## Example
The following uses the distilled 9B model to generate a photorealistic hummingbird image in a small number of steps:

```sh
mflux-generate-flux2 \
  --model flux2-klein-9b \
  --prompt "Photorealistic close-up of a hummingbird hovering near red flowers, frozen wings, detailed feathers, soft green background bokeh, high shutter speed look." \
  --steps 4 \
  --seed 640563507 \
  --width 1024 \
  --height 560
```

## Base model example
Base (non-distilled) FLUX.2 models use more steps and allow guidance > 1.0:

> [!WARNING]
> Base (non-distilled) FLUX.2 Klein models are typically slower and worse for general image editing, but can be successfully used for training.

```sh
mflux-generate-flux2 \
  --model flux2-klein-base-9b \
  --prompt "A red fox resting in fresh snow under soft winter light, detailed fur, gentle bokeh, natural color grading." \
  --steps 50 \
  --guidance 1.5 \
  --seed 640563507 \
  --width 1024 \
  --height 560
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein

model = Flux2Klein(model_config=ModelConfig.flux2_klein_9b())
image = model.generate_image(
    seed=640563507,
    prompt="Photorealistic close-up of a hummingbird hovering near red flowers, frozen wings, detailed feathers, soft green background bokeh, high shutter speed look.",
    num_inference_steps=4,
    width=1024,
    height=560,
)
image.save("hummingbird.png")
```
</details>

## Image-conditioned editing
FLUX.2 supports image-conditioned editing with one or more reference images:

![FLUX.2 Klein Edit Example](../../assets/flux2_klein_edit.jpg)

*Example image from [Unsplash](https://unsplash.com/photos/shallow-focus-photography-of-woman-outdoor-during-day-rDEOVtE7vOs)*

```sh
mflux-generate-flux2-edit \
  --model flux2-klein-9b \
  --image-paths person.jpg glasses.jpg \
  --prompt "Make the woman wear the eyeglasses (regular glasses, not sunglasses)" \
  --steps 4 \
  --seed 42
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2KleinEdit

model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_9b())
image = model.generate_image(
    seed=42,
    prompt="Make the woman wear the eyeglasses (regular glasses, not sunglasses)",
    image_paths=["person.jpg", "glasses.jpg"],
    num_inference_steps=4,
)
image.save("flux2_edit.png")
```
</details>

### KV-cache editing (`flux2-klein-9b-kv`)
For image-conditioned editing with reference images, [FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) is a 9B variant tuned for KV-cache inference. Select the model with `--model`; KV caching is enabled automatically when `--image-paths` provides reference tokens to cache. No extra CLI flag is required.

Use the same `mflux-generate-flux2-edit` command as above, swapping the model name:

```sh
mflux-generate-flux2-edit \
  --model flux2-klein-9b-kv \
  --image-paths person.jpg glasses.jpg \
  --prompt "Make the woman wear the eyeglasses (regular glasses, not sunglasses)" \
  --steps 4 \
  --seed 42
```

KV caching speeds up later denoising steps by reusing reference-image attention state from the first step. It only applies when reference images are provided via `--image-paths`.

> [!WARNING]
> Note: FLUX.2-klein-4B requires downloading the `black-forest-labs/FLUX.2-klein-4B` model weights (~15GB), FLUX.2-klein-9B requires `black-forest-labs/FLUX.2-klein-9B` model weights (~32GB), and FLUX.2-klein-9b-kv requires `black-forest-labs/FLUX.2-klein-9b-kv` model weights. Use quantization for smaller sizes.

## Notes
- FLUX.2 does not support `--negative-prompt`.
- Distilled FLUX.2 variants use `--guidance 1.0`. Base variants support guidance values above `1.0`.
- Supported distilled variants: `flux2-klein-4b` (default), `flux2-klein-9b`, and `flux2-klein-9b-kv` (KV-cache edit). Distilled models run in fewer steps than base.

## Training
Training also supports `flux2-klein-base-4b` and `flux2-klein-base-9b`.

### Fine-tuning
Use `mflux-train` with a training config. For the data/images folder layout, see the common [training docs](../common/README.md#training-lora).

Example (base model defaults to 50 steps):

```json
{
  "model": "flux2-klein-base-9b",
  "data": "images/",
  "seed": 42,
  "steps": 40,
  "guidance": 1.0,
  "quantize": null,
  "low_ram": false,
  "max_resolution": 1024,
  "training_loop": { "num_epochs": 100, "batch_size": 1, "timestep_low": 25, "timestep_high": 40 },
  "optimizer": { "name": "AdamW", "learning_rate": 1e-4 },
  "checkpoint": { "output_path": "train", "save_frequency": 25 },
  "monitoring": {
    "plot_frequency": 1,
    "generate_image_frequency": 20
  },
  "lora_layers": {
    "targets": [
      { "module_path": "transformer_blocks.{block}.attn.to_q", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.to_k", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.to_v", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.to_out", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.add_q_proj", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.add_k_proj", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.add_v_proj", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.to_add_out", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.ff.linear_in", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.ff.linear_out", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.ff_context.linear_in", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.ff_context.linear_out", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj", "blocks": { "start": 0, "end": 20 }, "rank": 16 },
      { "module_path": "single_transformer_blocks.{block}.attn.to_out", "blocks": { "start": 0, "end": 20 }, "rank": 16 }
    ]
  }
}
```

Run training:

```sh
mflux-train --config /path/to/train.json
```

### Edit fine-tuning (image-conditioned)
 The config looks identical for edit-based training and only differs in how the training data is prepared and named. For edit-style training (image in + prompt → image out), use auto-discovery with paired `*_out.*` and `*_in.*` files plus `*_in.txt` prompts, see the [training docs](../common/README.md#training-lora) for examples. Edit training is supported for Flux2 Klein base models. Preview prompts come from `data/preview*.txt` and the preview images are `data/preview*.{png,jpg,jpeg,webp}`.
