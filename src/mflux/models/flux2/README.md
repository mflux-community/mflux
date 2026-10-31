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

You can also call the steps of both commands from Python, for example to wrap them in a plugin. `Flux2Command` in `mflux.models.flux2.cli.flux2_generate` runs `mflux-generate-flux2`, and `Flux2EditCommand` in `mflux.models.flux2.cli.flux2_edit_generate` runs `mflux-generate-flux2-edit`. Each has `validate(args)`, `load(args)` and `generate(model, args, seed, prompt)`. They work like the Z-Image Turbo steps; the [Z-Image README](../z_image/README.md#z-image-turbo-example) has a full script and the rules for keeping a model loaded. `validate()` loads no weights (the edit command does not open the input images) and returns the model config; it raises `ValueError` for a refused flag, such as `--guidance` on a distilled model, and `ModelConfigError` for a `--model` outside the klein family. `load()` runs it first. `generate()` uses guidance 1.0 when `--guidance` is not set.

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

## Small text encoder (experimental)
`--small-text-encoder` replaces klein-4B's Qwen3-4B text encoder with `Qwen/Qwen3-0.6B` plus a 220M adapter (`AiArtLab/qwen3-0.6b-4b-adapter`) that maps its hidden states into klein's 7680-dim conditioning space. The transformer, VAE and tokenizer stay klein's, so the klein `text_encoder` weights are never downloaded.

```sh
mflux-generate-flux2 \
  --model flux2-klein-4b \
  --small-text-encoder \
  --prompt "A red fox walks through a snowy forest at dusk." \
  --steps 4 \
  -q 8
```

- Works with `flux2-klein-4b` and `flux2-klein-base-4b` (`mflux-generate-flux2` and `mflux-generate-flux2-edit`). Other models exit with an error.
- Prompts are capped at 256 tokens (the native encoder takes 512). Fine details can drift from the native encoder.
- `-q` quantizes the 0.6B encoder like the rest of the model; the adapter always stays float32.
- A model built with the flag cannot be saved (`save_model()` raises), and the flag is recorded in the image metadata so `--config-from-metadata` replays it.
- Measured on FLUX.2-klein-4B (Apple M4 Pro, 64 GB): the loaded weights drop from 14.9 GB to 9.3 GB in bf16, from 8.0 GB to 5.4 GB with `-q 8` and from 4.3 GB to 3.3 GB with `-q 4` (the float32 adapter is 0.8 GB of that). With a saved 4-bit model at 512x512, peak memory drops from 4.3 GB to 3.7 GB. With `--low-ram` at 1024x1024 the peak is the same, because the denoising and VAE decode set it. A 1024x1024 image in bf16 (4 steps, prompt encoding included) takes about 20 s instead of 27 s.
- The first run downloads about 2.3 GB (`Qwen/Qwen3-0.6B` and the adapter) instead of the 7.5 GB klein `text_encoder`.
- Danbooru-style tag prompts that give an anime image with the native encoder can come out photographic with the adapter (the diffusers reference does the same).
- Attribution: adapter by AiArtLab ([AiArtLab/qwen3-0.6b-4b-adapter](https://huggingface.co/AiArtLab/qwen3-0.6b-4b-adapter), Apache-2.0); [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) is Apache-2.0.

## Small decoder
[FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) (Apache-2.0) is a distilled VAE decoder from Black Forest Labs. It has 28M parameters instead of 50M. Add `--small-decoder` to `mflux-generate-flux2` or `mflux-generate-flux2-edit` to use it. Black Forest Labs lists it for all FLUX.2 klein models:

```sh
mflux-generate-flux2 \
  --prompt "A red fox walks through a snowy forest at dusk." \
  --small-decoder \
  --seed 42
```

- The first run downloads about 250 MB. The transformer, the text encoder and the VAE encoder do not change.
- With the default [strip decode](#vae-decode), the small decoder needs 1.1 s instead of 1.3 s at 1024x1024. It needs 1.2 GB instead of 1.6 GB. At 2048x2048 it takes 4.7 s instead of 5.5 s and needs 4.3 GB instead of 5.6 GB. These are VAE-only decodes of klein-9B-kv `-q 4` latents on an Apple Silicon Mac with 36 GB.
- The peak of a whole `-q 4 --low-ram` run does not decrease, because the decode no longer sets it. With klein-9B-kv it is 7.2 GB instead of 7.1 GB at 1024x1024, and 10.4 GB instead of 10.3 GB at 2048x2048.
- With `--vae-full-precision` the decode sets the peak again, and the small decoder saves more. The decode needs 5.5 GB instead of 7.3 GB at 1024x1024, and 21.5 GB instead of 28.5 GB at 2048x2048.
- The images are almost the same: 31-41 dB PSNR against the standard decoder on our test prompts.
- `--pid-decode` replaces the VAE decoder, so the two flags cannot be used together.
- The image metadata records the flag, and `--config-from-metadata` uses it again.
- `save_model()` refuses a model loaded with `small_decoder=True`. Save the model without the flag.

## VAE decode
The VAE decodes the image in bf16 row strips of 128 rows. Each GroupNorm uses statistics from the full image, so the strips do not change the colours, and the image has no seams. The decoder releases the temporary data of each block before the next block starts, and the attention runs in chunks of 4096 queries.

- `--vae-full-precision` decodes each block on the full image, as the earlier decode did. The metadata records the flag, so `--config-from-metadata` uses it again.
- `--vae-tiling` and `--vae-tile-size` do not change the decode, and the CLI shows a warning. They still tile the encode of an input image.
- `--low-ram` still releases the text encoder and the transformer before the decode. It does not change the decode.
- `--small-decoder` uses the same strip decode.

Measured as a decode of saved latents, with only the VAE loaded. Test setup: Apple Silicon Mac, 36 GB, MLX 0.32.0, klein-9B-kv `-q 4`, seed 42, 14 prompts at 1024x1024 and 3 prompts at 2048x2048.

| Decode | Peak 1024x1024 | Peak 2048x2048 | Time 1024x1024 | PSNR vs earlier decode |
|---|---|---|---|---|
| Earlier full-image decode | 10.3 GB | 40.5 GB | 0.87 s | – |
| Earlier `--vae-tiling` | 3.1 GB | – | 0.99 s | 25–28 dB (visible colour shift) |
| Default (strips) | 1.6 GB | 5.6 GB | 1.27 s | 43–51 dB |
| `--vae-full-precision` | 7.3 GB | 28.5 GB | 0.96 s | identical on the test Mac |

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
  "gradient_checkpointing": false,
  "max_resolution": 1024,
  "training_loop": { "num_epochs": 100, "batch_size": 1, "timestep_low": 25, "timestep_high": 40 },
  "optimizer": { "name": "AdamW", "learning_rate": 1e-4 },
  "checkpoint": { "output_path": "train", "save_frequency": 25 },
  "monitoring": {
    "plot_frequency": 20,
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

> [!TIP]
> On a 32–36 GB Mac, set `"gradient_checkpointing": true` (and `"quantize": 8`); without it the first training step of an edit run needs more memory than the machine has and the process is killed. See the common [training docs](../common/README.md#training-lora).

### Edit fine-tuning (image-conditioned)
 The config looks identical for edit-based training and only differs in how the training data is prepared and named. For edit-style training (image in + prompt → image out), use auto-discovery with paired `*_out.*` and `*_in.*` files plus `*_in.txt` prompts, see the [training docs](../common/README.md#training-lora) for examples. Edit training is supported for Flux2 Klein base models. Preview prompts come from `data/preview*.txt` and the preview images are `data/preview*.{png,jpg,jpeg,webp}`.
