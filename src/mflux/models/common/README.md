## Common CLI patterns (shared across models)

This README covers stable, shared patterns. For model-specific usage, see each model README and the model list in the top-level `README.md`.

> [!NOTE]
> MFLUX supports many CLI combinations. The quickest way to get a correct command is to ask a coding agent (Cursor, Claude Code, or similar), e.g. “Can you help me generate an image using z-image?” When in doubt, check the current CLI entrypoints for up-to-date support.

### Quick navigation

- [Quantization](#quantization)
- [Local model paths](#local-model-paths)
- [Third-party HuggingFace models](#third-party-huggingface-models)
- [Prompt files](#prompt-files)
- [LoRA basics](#lora-basics)
- [Training (LoRA)](#training-lora)
- [Output files](#output-files)
- [Auto seeds](#auto-seeds)
- [Metadata reuse](#metadata-reuse)
- [Metadata inspection](#metadata-inspection)
- [PiD pixel-diffusion decode](#pid-pixel-diffusion-decode)
- [Step reuse (step cache)](#step-reuse-step-cache)
- [Float16 compute (Apple M1 family)](#float16-compute-apple-m1-family)
- [Resource and inspection options](#resource-and-inspection-options)
- [MLX cache limit](#mlx-cache-limit)
- [Cache locations](#cache-locations)
- [Capabilities by task](#capabilities-by-task)
- [Command reference](#command-reference)
- [CLI completions](#cli-completions)

---

## Quantization

Quantization reduces memory use and speeds up inference. Most models support `--quantize` (3, 4, 5, 6, 8). This is the on-the-fly option, and quantization is performed as weights are loaded.

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --quantize 8 \
  --steps 9 \
  --prompt "A photo of a dog"
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8, model_config=ModelConfig.z_image_turbo())
image = model.generate_image(
    seed=42,
    prompt="A photo of a dog",
    num_inference_steps=9,
)
image.save("dog.png")
```
</details>

### Saving a quantized model

```sh
mflux-save \
  --path "/Users/me/models/z-image-turbo_8bit" \
  --model z-image-turbo \
  --quantize 8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8, model_config=ModelConfig.z_image_turbo())
model.save_model("/Users/me/models/z-image-turbo_8bit")
```
</details>

You can optionally include LoRAs when saving:

```sh
mflux-save \
  --path "/Users/me/models/z-image-turbo_8bit" \
  --model z-image-turbo \
  --quantize 8 \
  --lora-paths "/path/to/lora.safetensors" \
  --lora-scales 0.7
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(
    quantize=8,
    model_config=ModelConfig.z_image_turbo(),
    lora_paths=["/path/to/lora.safetensors"],
    lora_scales=[0.7],
)
model.save_model("/Users/me/models/z-image-turbo_8bit")
```
</details>

> [!WARNING]
> Quantized models saved with mflux < v0.6.0 are incompatible with newer versions. Re-save using `mflux-save`.

---

## Local model paths

Use a local path with `--model` and specify `--base-model` when required:

```sh
mflux-generate-z-image-turbo \
  --model "/Users/me/models/z-image-turbo" \
  --base-model z-image-turbo \
  --steps 9 \
  --prompt "Luxury food photograph"
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(
    model_config=ModelConfig.z_image_turbo(),
    model_path="/Users/me/models/z-image-turbo",
)
image = model.generate_image(
    seed=42,
    prompt="Luxury food photograph",
    num_inference_steps=9,
)
image.save("luxury_food.png")
```
</details>

Local HuggingFace-style weights should match the repo layout for the model. See the model README if you need the exact directory structure.

---

## Third-party HuggingFace models

The `--model` argument can be a HuggingFace repo or a local path:

```sh
mflux-generate-z-image-turbo \
  --model filipstrand/Z-Image-Turbo-mflux-4bit \
  --steps 9 \
  --prompt "A beautiful landscape"
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(
    model_config=ModelConfig.z_image_turbo(),
    model_path="filipstrand/Z-Image-Turbo-mflux-4bit",
)
image = model.generate_image(
    seed=42,
    prompt="A beautiful landscape",
    num_inference_steps=9,
)
image.save("landscape.png")
```
</details>

---

## Prompt files

Use `--prompt-file` to iterate on prompts without restarting the CLI:

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --prompt-file ./prompt.txt
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

prompt = Path("./prompt.txt").read_text().strip()
model = ZImageTurbo(model_config=ModelConfig.z_image_turbo())
image = model.generate_image(
    seed=42,
    prompt=prompt,
    num_inference_steps=9,
)
image.save("prompt_file.png")
```
</details>

---

## LoRA basics

By default, LoRA and LoKr weights are **baked** into the base model after load (one-time merge) so inference uses a single matmul per layer. With `--quantize` on FLUX, FLUX.2, Qwen-Image, Krea 2 and Z-Image, a checkpoint that stores full-precision weights gets the adapter added to those weights, in float32, before they are quantized. Elsewhere, and whenever the checkpoint is already quantized, the adapter is merged into the dequantized weights and the result is re-quantized. A weak adapter at a low scale loses part of its effect that way: one Z-Image LoRA at scale 0.5 came out at 80% of its strength on q8 ([#814](https://github.com/mflux-community/mflux/issues/814)). Use `--no-bake-lora` to keep runtime adapter layers: they apply the adapter in full on any checkpoint, they cost speed and memory, and they let you change scales programmatically without reloading.

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --prompt "a portrait" \
  --lora-paths "/local/path/to/lora.safetensors" \
  --lora-scales 0.8
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(
    model_config=ModelConfig.z_image_turbo(),
    lora_paths=["/local/path/to/lora.safetensors"],
    lora_scales=[0.8],
    bake_lora=True,
)
image = model.generate_image(
    seed=42,
    prompt="a portrait",
    num_inference_steps=9,
)
image.save("portrait_lora.png")
```
</details>

You can also use a real HuggingFace LoRA repo with Z-Image Turbo:

```sh
mflux-generate-z-image-turbo \
  --model filipstrand/Z-Image-Turbo-mflux-4bit \
  --steps 9 \
  --prompt "t3chnic4lly vibrant 1960s portrait" \
  --lora-paths renderartist/Technically-Color-Z-Image-Turbo \
  --lora-scales 0.5
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(
    model_config=ModelConfig.z_image_turbo(),
    model_path="filipstrand/Z-Image-Turbo-mflux-4bit",
    lora_paths=["renderartist/Technically-Color-Z-Image-Turbo"],
    lora_scales=[0.5],
)
image = model.generate_image(
    seed=42,
    prompt="t3chnic4lly vibrant 1960s portrait",
    num_inference_steps=9,
)
image.save("portrait_hf_lora.png")
```
</details>

For multi-LoRA, pass multiple paths and scales. For library usage, set `LORA_LIBRARY_PATH` and pass basenames.

DoRA adapters saved with `lora_A` / `lora_B` matrices (the PEFT layout, with `lora_magnitude_vector` or `dora_scale`) are not supported for any model. mflux stops with an error instead of loading them without their magnitude. LoKr files with `dora_scale` still load (see below).

### LyCORIS LoKr (FLUX.1 and FLUX.2)

LyCORIS LoKr safetensors use the same `--lora-paths` / `lora_scales` API as classic LoRA. mflux accepts direct `lokr_w1` / `lokr_w2` tensors, factorized `lokr_w1_a` / `lokr_w1_b` (and `lokr_w2_*`, optional `lokr_t2`), optional `dora_scale`, and common ComfyUI / SimpleTuner key prefixes (`lycoris_*`, `lora_unet_*`, `diffusion_model.*`).

```sh
mflux-generate-flux2 \
  --model flux2-klein-9b \
  --prompt "a portrait" \
  --steps 4 \
  --lora-paths "/path/to/lycoris-lokr.safetensors" \
  --lora-scales 1.0
```

FLUX.1 uses `mflux-generate` with `--model dev` (or `schnell` / `krea-dev`) and the same LoKr flags.

---

## Training (LoRA)

![Training loss preview](../../assets/train_loss.png)

- Use `mflux-train --config /path/to/train.json` to start a run.
- Resume with `mflux-train --resume /path/to/checkpoint.zip`.
- Training supports normal (txt2img) and edit modes; the mode is auto-detected from your data folder layout (see examples below).
- Expected layout: a config file plus a data folder with images and optional prompt files.
- **Edit mode limitation**: Edit training currently supports only FLUX.2-klein-base models (`flux2-klein-base-4b` or `flux2-klein-base-9b`).
- **Memory**: `"gradient_checkpointing": true` recomputes each transformer block's activations during the backward pass instead of keeping all of them, and evaluates the gradients last-block-first (the evaluation order matters in MLX; in tree order the recomputed blocks all stay alive and the saving disappears). Steps get ~30% slower and peak memory drops to roughly "weights + a couple of blocks". Turn it on when a run is killed at the first training step; it is what makes FLUX.2 klein edit training fit on 32–36 GB machines. Combine with `"quantize": 8` and `"low_ram": true` on the tightest machines.
- To train a local copy of a model, you can specify the path to that model in the training spec JSON alongside the `model` using the `model_path` key.
- For model-specific config fields, see the [Z-Image README](../z_image/README.md) and [FLUX.2 README](../flux2/README.md).

CLI example using the repo training sample ([train.json](training/_example/train.json)):

```sh
uv run mflux-train --config src/mflux/models/common/training/_example/train.json
```

Example dataset layout (txt2img) — point `data` in the training config at a folder with this kind of structure of (text, image) pairs:

```
01.txt
01.png
02.txt
02.png
03.txt
03.jpg
04.txt
04.jpg
preview_1.txt   <-- (optional) preview prompt run periodically during training
preview_2.txt   <-- (optional) preview prompt run periodically during training
```

Preview prompts are read from `data/preview*.txt` when monitoring is enabled. If no `preview*.txt` files are found, the first data prompt is used as a fallback (for both txt2img and edit training). For edit training, if preview prompts are explicitly provided (not fallback), each preview prompt must have a matching `data/preview*.{png,jpg,jpeg,webp}` image.

The repo sample includes images like [01.jpeg](training/_example/images/01.jpeg). The prompts in the example images were auto-generated by the Qwen3-VL-235B-A22B model from [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm)


Example dataset layout (edit) — point `data` in the training config at a folder with this kind of structure of (text_in, image_in, image_out) triplets:

```
01_in.txt
01_in.png
01_out.png
02_in.txt
02_in.png
02_out.png
03_in.txt
03_in.png
03_out.png
preview_1.txt  <-- (optional) preview prompts run periodically during training
preview_1.png  <-- (optional) edit preview image for preview_1.txt
preview_2.txt  <-- (optional) preview prompts run periodically during training
preview_2.png  <-- (optional) edit preview image for preview_2.txt
```

We use `_in` for the context/reference image and `_out` for the target image to make the direction explicit and keep pairing unambiguous.
Preview text always comes from `data/preview*.txt`. For edit training, if preview prompts are explicitly provided (not using fallback), each preview prompt must have a matching `data/preview*.{png,jpg,jpeg,webp}` image. Non-edit training does not accept preview images.

### Training outputs

Training output folders include checkpoints, loss plots, and preview images:
```
checkpoints/
  0000030_checkpoint.zip
loss/
  loss.html
preview/
  0000000_preview_image_preview.png
  0000015_preview_image_preview.png
  0000030_preview_image_preview.png
```

Each checkpoint zip (e.g., `0000030_checkpoint.zip`) is a self-contained snapshot of the training state, but the original data folder must still be available to resume training.

---

## Output files

Use `--output` to set the output filename. When generating multiple seeds or images, MFLUX appends suffixes automatically.
```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --prompt "a portrait" \
  --output ./outputs/zimage.png
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(model_config=ModelConfig.z_image_turbo())
image = model.generate_image(
    seed=42,
    prompt="a portrait",
    num_inference_steps=9,
)
image.save("./outputs/zimage.png")
```
</details>

---

## Auto seeds

Use `--auto-seeds` to generate multiple random seeds in one run:

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --auto-seeds 4 \
  --prompt "a portrait"
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(model_config=ModelConfig.z_image_turbo())
seeds = [42, 1337, 2024, 9999]
for seed in seeds:
    image = model.generate_image(
        seed=seed,
        prompt="a portrait",
        num_inference_steps=9,
    )
    image.save(f"portrait_{seed}.png")
```
</details>

---

## Metadata reuse

Run a generation with `--make-conf` to write a JSON sidecar of the generation parameters next to the image. Then use those parameters again in a subsequent run. The sidecar filename uses the `.metadata.json` suffix. To stop the image from holding embedded metadata (EXIF), add `--no-exif`. The old names `--metadata`, `--no-metadata` and `--config-from-metadata` continue to work as aliases:

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --make-conf \
  --output ./image.png
```

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --config-from-conf ./image.metadata.json \
  --prompt "Same composition, warmer light"
```

Anything the sidecar recorded is restored — model, prompt and negative prompt, seed, steps, guidance, quantization, dimensions, LoRAs, and the init images of the edit CLIs — and anything named on the command line wins over it, option by option. Since the sidecar can supply the init images, `mflux-generate-qwen-edit` and `mflux-generate-flux2-edit` accept `--config-from-conf` on its own, without `--image-paths`.

<details>
<summary>Python API</summary>

```python
import json

from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

with open("./image.metadata.json", "r") as handle:
    metadata = json.load(handle)

model = ZImageTurbo(model_config=ModelConfig.z_image_turbo())
image = model.generate_image(
    seed=metadata.get("seed", 42),
    prompt="Same composition, warmer light",
    num_inference_steps=metadata.get("steps", 9),
    width=metadata.get("width", 1024),
    height=metadata.get("height", 1024),
)
image.save("image_warmer_light.png")
```
</details>

Anything given on the command line wins over the sidecar, adapters included: `--lora` (or `--lora-paths`) replaces the LoRAs the sidecar recorded rather than adding to them, and `--lora-paths` with no values re-runs without any. `--lora-scales` on its own is the exception — it keeps the sidecar's adapters and applies the new strengths, for a re-run of the same LoRA at a different weight.

A sidecar records each adapter's resolved absolute path, so one copied from another machine can name a file that is not there. That is an error rather than a silent no-op, since the run cannot reproduce the image without the adapter: pass `--lora` to point at your own copy, or `--lora-paths` with no values to run without it.

---

## Metadata inspection

Use `mflux-info` to inspect metadata embedded in an image generated by MFLUX:

```sh
mflux-info ./image.png
```

<details>
<summary>Python API</summary>

```python
from mflux.utils.metadata_reader import MetadataReader

metadata = MetadataReader.read_all_metadata("./image.png")
print(metadata)
```
</details>

---

## PiD pixel-diffusion decode

`--pid-decode` replaces the VAE decoder with [NVIDIA PiD](https://huggingface.co/nvidia/PiD), a 4-step distilled diffusion that denoises in pixel space conditioned on the VAE latent and the prompt. The output is **4× the generation size**: `--width 512 --height 512` writes a 2048×2048 image, and the extra detail is generated rather than interpolated.

First use downloads two Hugging Face repos, ~8 GB combined: `nvidia/PiD` and `google/gemma-2-2b-it` (**gated** — accept the Gemma licence on the Hub, then `hf auth login`).

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 8 \
  --prompt "a portrait" \
  --width 832 --height 1248 \
  --pid-decode
```

| Flag | Effect |
| --- | --- |
| `--pid-decode` | Decode with PiD instead of the VAE. Output is 4× the generation size. |
| `--pid-degrade-sigma S` | Noise the latent to flow-matching sigma `S` (0.0–0.8) before decoding. See below. |

**It is not a decode-faithful operation.** PiD finishes the image in pixel space at 4×, so it re-draws rather than sharpens: fine geometry can shift (bird beak plates, small typography, jewellery) relative to a VAE decode of the same latent at the same seed. It is a better-looking image of the same scene, not the same image at higher resolution.

**Skin is the weak spot.** On portraits PiD invents blemishes on large smooth expanses — moles that grow, speckle on a chest or shoulder, added skin age on a face. Feathers, fur, fabric and engraved metal are where it looks best.

**`--pid-degrade-sigma` is not a fix for that** — it is the knob PiD's own reference exposes (`from_clean.py` forward-noises the latent by sigma and passes the same sigma to the LQ gates; single-image mode defaults to 0.0), and it buys a *different* sample, not a cleaner one. Measured on Krea 2 at 512×640, sigma 0.0 → 0.1 → 0.2: high-frequency energy on the face rises 54× → 82× → 99× relative to a VAE decode (PiD invents *more*, not less), every channel darkens monotonically (−4 to −11 levels by 0.2), and framing drifts enough to reframe a fixed crop. On smooth skin it changes which blemishes appear rather than how many. Leave it at 0.0 unless you are deliberately sampling for variety.

**On Ideogram 4**, a plain-text prompt can trip the model's own safety filter and return a blank page. That happens on a normal VAE decode too — it is Ideogram's behaviour, not PiD's. Use a structured JSON caption.

Metadata records the PiD flags, so `--config-from-conf` reproduces a PiD run. `config.height/width` stay the *generation* dimensions, which is what reproduces the run — the file on disk is 4× larger.

---

## Step reuse (step cache)

Commands that support it take `--step-cache-ratio` (TeaCache-style step reuse). On that fraction of denoise steps the transformer is skipped and the previous step's prediction is reused, while the scheduler still takes its normal step. The skipped steps are the ones whose timestep signal changes least. The first and last 10% of the run always run, and runs under 10 steps are unaffected, so this only pays off on models sampled for many steps. It trades a little detail for speed: the ratio is recorded in image metadata and replayed by `--config-from-conf`. Check `mflux-capabilities` for the commands that honor it; today that is `mflux-generate-qwen-2.1`.

```sh
mflux-generate-qwen-2.1 --prompt "a lighthouse at dusk" --steps 40 --step-cache-ratio 0.25
```

<details>
<summary>Adding step reuse to a model</summary>

The selection and reuse logic lives in `mflux.models.common.step_cache.StepCache` and is model-agnostic. By default steps are scored by their sigma, which every flow-match scheduler exposes. A model can pass `signal_fn` to score with a richer signal, such as its timestep-embedding MLP (Qwen Image 2.1 does this). Wire it into the denoise loop and add `parser.add_step_cache_arguments()` to the CLI:

```python
step_cache = StepCache.for_run(config, ratio=step_cache_ratio, signal_fn=self.transformer.time_text_embed)
for t in config.time_steps:
    noise = step_cache.reuse(t)
    if noise is None:
        noise = ...  # the model call, including any guidance pass
        step_cache.store(noise)
    latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
```

Pass `generation_parameters=StepCache.generation_parameters(step_cache_ratio)` to `ImageUtil.to_image` so the ratio lands in metadata. Samplers that keep their own history across steps (multistep solvers) need a review before reuse is enabled, and each model should be quality-checked against its uncached output.

</details>

---

## Float16 compute (Apple M1 family)

`--compute-precision float16` runs the inside of the attention and feed-forward layers in float16. The residual stream, the AdaLN modulation, the norms between those layers, the text encoder and the VAE keep the model's own precision. It is meant for GPUs without native bfloat16, the Apple M1 family, where MLX's matmul and attention kernels are fastest in float16. Without the flag nothing changes. With it the image changes slightly for the same seed, so the image metadata records the flag and `--config-from-conf` replays it. Commands that support it: `mflux-generate-flux2`, `mflux-generate-flux2-edit`, `mflux-generate-z-image`, `mflux-generate-z-image-turbo`, `mflux-generate-qwen-2.1` and `mflux-generate-qwen-2.1-edit` (`mflux-capabilities` lists them).

```sh
mflux-generate-z-image-turbo --prompt "A puffin standing on a cliff" --steps 9 -q 4 --compute-precision float16
```

Measured on an M1 Max (64 GB) at 1024×1024. PSNR compares each image with the one the same seed gives without the flag (three seeds; two for Qwen-Image 2.1):

| model | s/step, without → with | total time | PSNR |
|---|---|---:|---|
| FLUX.2 klein 4B, `-q 4` | 6.8 → 5.0 | −22% | 38–48 dB |
| Z-Image Turbo, `-q 4` | 10.7 → 7.5 | −27% | 36–37 dB |
| FLUX.2 klein 9B, unquantized | 11.6 → 10.5 | −4% to −12% | 30–43 dB |
| Qwen-Image 2.1 edit, 6-step LoRA, unquantized | 12.6 → 10.8 | −12% | 44–48 dB |

The gain depends on the chip. Reported in the pull request discussion: an M2 Ultra ran 18% to 22% faster in total, and an M5 Max showed no change. On an M4 Max (36 GB), Z-Image Turbo `-q 4` took 4% to 6% less time for each step. FLUX.2 klein 9B KV and Qwen-Image 2.1 changed by less than the run-to-run noise.

`--float32` exists only on the Z-Image commands. The FLUX.2 and Qwen-Image 2.1 commands have no `--float32` option.

<details>
<summary>Python API</summary>

```python
import mlx.core as mx

from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=4, compute_precision=mx.float16)
image = model.generate_image(
    seed=42,
    prompt="A puffin standing on a cliff",
    num_inference_steps=9,
)
image.save("puffin.png")
```

`Flux2Klein`, `Flux2KleinEdit`, `QwenImage21` and `QwenImage21Edit` take the same `compute_precision` argument. The weights of those layers are cast in place when the model loads, after quantization and LoRA. For that reason `save_model()` raises an error on a model built with `compute_precision`: build the model without it to save it.
</details>

<details>
<summary>Adding it to a model</summary>

The logic lives in `mflux.models.common.compute_precision.ComputePrecision`. Give each attention and feed-forward module a `self.compute_precision = ComputePrecision()`, pass its input through `to_compute` and its output through `to_stream` together with the input dtype, and add a transformer method that calls `precision.apply(self, (YourAttention, YourFeedForward))`. The initializer calls that method after weights and LoRA are applied, so quantization scales and LoRA factors are cast too. It also stores the precision as `model.compute_precision`, and `generate_image` passes `generation_parameters=self.compute_precision.generation_parameters()` to `ImageUtil.to_image` so the image metadata records it. If the activations inside a module exceed the float16 range, as in Z-Image, scale them with `shrink(x, headroom)` and pass the same headroom to `to_stream`. Then add `parser.add_compute_precision_arguments()` to the command.

</details>

---

## Resource and inspection options

Use `--low-ram` to reduce memory usage (at the cost of performance). Use `--stepwise-image-output-dir` to save stepwise images for inspection.

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --prompt "a portrait" \
  --low-ram \
  --stepwise-image-output-dir ./steps
```

<details>
<summary>Python API</summary>

```python
from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo
from mflux.models.z_image.latent_creator import ZImageLatentCreator

model = ZImageTurbo(model_config=ModelConfig.z_image_turbo())
model.callbacks.register(BatterySaver(battery_percentage_stop_limit=20))
model.callbacks.register(MemorySaver(model=model, keep_transformer=False))
model.callbacks.register(
    StepwiseHandler(
        model=model,
        output_dir="./steps",
        latent_creator=ZImageLatentCreator,
    )
)
image = model.generate_image(
    seed=42,
    prompt="a portrait",
    num_inference_steps=9,
)
image.save("image.png")
```
</details>

### Logging

CLIs are configured with default logging. Pass `-v`/`--verbose` to add more debug output.
Logs go to stderr, so piping stdout is unaffected.
When [rich](https://github.com/Textualize/rich) is installed and stderr is a terminal, logs become even prettier.

---

## MLX cache limit

Use `--mlx-cache-limit-gb` to cap MLX cache usage.

```sh
mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --prompt "a portrait" \
  --mlx-cache-limit-gb 2.5
```

- Value must be positive (`> 0`).
- The value is converted internally using decimal gigabytes (`GB * 1000^3`).
- Works in both normal mode and `--low-ram` mode.

---

## Cache locations

- **MFLUX cache**: set `MFLUX_CACHE_DIR` to override the default (`~/Library/Caches/mflux` on macOS, `~/.cache/mflux` on Linux).
- **Hugging Face cache**: set `HF_HOME` (for example, `HF_HOME=/Volumes/T7/.cache/huggingface`). For more details, refer to the Hugging Face cache documentation.

Example with `HF_HOME` set for a generation run:

```sh
HF_HOME=/Volumes/T7/.cache/huggingface \
  mflux-generate-z-image-turbo \
  --model z-image-turbo \
  --steps 9 \
  --prompt "A moody coastal landscape at dusk"
```

<details>
<summary>Python API</summary>

```python
import os

from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImageTurbo

os.environ["HF_HOME"] = "/Volumes/T7/.cache/huggingface"
model = ZImageTurbo(model_config=ModelConfig.z_image_turbo())
image = model.generate_image(
    seed=42,
    prompt="A moody coastal landscape at dusk",
    num_inference_steps=9,
)
image.save("coastal_dusk.png")
```
</details>

---

## Capabilities by task

- Text-to-image and image-to-image generation
- Editing: in-context edit, multi-image edit, virtual try-on
- ControlNet, depth conditioning, fill/inpainting, Redux
- Upscaling (SeedVR2 and Flux ControlNet)
- Quantization, local paths, and third-party models
- LoRA support and LoRA library tooling
- Metadata export and reuse
- Training (LoRA)
- Utilities: depth map extraction, FIBO VLM inspire/refine

---

## Command reference

Use the model READMEs for exact command names and options. General utilities that work across models:

- **Model management**: `mflux-save`, `mflux-info`, `mflux-lora-library`
- **Training**: `mflux-train`
- **Utilities**: `mflux-save-depth`

For model-specific generation commands (Flux, Flux2, Qwen, Z-Image, FIBO) and specialized features (ControlNet, depth, fill, Redux, in-context editing, kontext, concept attention, upscaling), see the respective model READMEs and the top-level `README.md`.

---

## CLI completions

MFLUX provides ZSH shell completions for all CLI commands. Install with:

```sh
mflux-completions
```

Then reload your shell (`exec zsh`) and enjoy tab completion for all mflux commands, options, and arguments. Completions include smart suggestions for model names, quantization levels, LoRA styles, and file paths.

For detailed installation instructions, troubleshooting, and advanced usage, see [`src/mflux/cli/completions/README.md`](../../cli/completions/README.md).
