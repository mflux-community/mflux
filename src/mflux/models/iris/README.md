# Iris-3B
This directory contains MFLUX's MLX implementation of **Iris-3B**.

MFLUX supports [Iris-3B Turbo](https://huggingface.co/speridlabs/iris-3b) (distilled, 4 steps) and [Iris-3B](https://huggingface.co/speridlabs/iris-3b) (the undistilled base model, 100 steps with guidance) from Speridlabs. It is a 3B-parameter diffusion transformer that generates directly in **pixel space**: there is no VAE and no latent space, so the image is predicted as RGB values and nothing is lost to a latent decoder. The text encoder is the language tower of Qwen3-VL-4B-Instruct (an 8 GB download of its own). Both models come from one Hugging Face repository, Turbo from its `turbo/` folder.

All the standard modes of mflux apply: quantization (`-q 4`, `-q 8`), `mflux-save`, `--low-ram`, metadata export and `--config-from-metadata`.

![Iris-3B Example](../../assets/iris_3b.jpg)

*Turbo, 1024×1024, 4 steps, seed 42: the fox prompt below.*

## Example
The following uses the distilled model (4 steps, no guidance) for a text-to-image generation:

```sh
mflux-generate-iris \
  --prompt "a red fox sleeping in fresh snow, golden hour" \
  --seed 42
```

The default size is 1024×1024. Width and height must be multiples of 16. Stay near one megapixel: Turbo was distilled
on 1024×1024, 752×1344, 896×1344, 896×1200 and their transposes ([model card](https://huggingface.co/speridlabs/iris-3b)),
and Iris works on pixels with no VAE to absorb a size it hasn't seen. At 512×512 both Turbo and the base model come out
broken up (blocks, stray text, smeared shapes); at 768×768 the same prompt is clean. For a smaller image, generate at
1024 and resize.

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.iris.variants import Iris

model = Iris(model_config=ModelConfig.iris_3b())
image = model.generate_image(
    seed=42,
    prompt="a red fox sleeping in fresh snow, golden hour",
    width=1024,
    height=1024,
)
image.save("fox.png")
```
</details>

## Base model example
The base model is undistilled. It runs 100 steps (a second-order DPM-Solver on the flow schedule) with classifier-free guidance (`--guidance` defaults to 3.0, `--negative-prompt` defaults to empty), so every step is two transformer passes. These are the defaults of the upstream implementation.

```sh
mflux-generate-iris \
  --model iris-3b-base \
  --prompt "a red fox sleeping in fresh snow, golden hour" \
  --seed 42
```

For Turbo `--steps`, `--guidance` and `--negative-prompt` have no effect and mflux prints a warning.

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.iris.variants import Iris

model = Iris(model_config=ModelConfig.iris_3b_base())
image = model.generate_image(
    seed=42,
    prompt="a red fox sleeping in fresh snow, golden hour",
    num_inference_steps=100,
    guidance=3.0,
    negative_prompt="",
)
image.save("fox_base.png")
```
</details>

## LoRAs
`--lora` loads Iris LoRAs with the names of the reference checkpoint: the kohya names the [reference repo](https://github.com/speridlabs/iris-3b) and its forks save (`blocks.0.q_proj.lora_down`), and the PEFT names under `diffusion_model.` that ai-toolkit saves. All 311 linear layers of the transformer can take a LoRA.

A LoRA stays a separate layer by default. Merged into the bfloat16 weights (`--bake-lora`), the rank-64 LoRA below moved the composition of the image (22 dB from the unmerged one, against 30 dB with `-q 8`, which merges in float32 before it quantizes). Unmerged, a step took about 10% longer.

Few-step LoRAs distilled from the base model, like [pentacoxian-dev/iris-3b-turbo-lora](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora), record `turbo_steps` and `shift` in their metadata. With such a LoRA the base model runs that sampler, as the [reference fork](https://github.com/pentaCoxian/iris-3b-turbo) does: 4 steps from pure noise on the shifted grid, one pass per step without CFG, and fresh noise between steps. mflux prints a line when it switches. A LoRA at scale 0 does not switch it, and `mflux-save` refuses a model with such a LoRA, since the saved model could not record the sampler. `--steps` still sets the count; `--guidance` and `--negative-prompt` have no effect, and mflux warns when they are given.

```sh
mflux-generate-iris \
  --model iris-3b-base \
  --lora pentacoxian-dev/iris-3b-turbo-lora:r2-1000/student.safetensors \
  --prompt "a red fox sleeping in fresh snow, golden hour" \
  --seed 42
```

On an M5 Max at 1024×1024 that takes about 1 second per step and 8 to 9 seconds for the whole run, loading included, against 2.6 seconds per step for the base model with CFG.

## Training
`mflux-train` trains a LoRA on the base model, for text to image. Turbo is distilled, so it is not trained. For the folder of images and captions, see the common [training docs](../common/README.md#training-lora).

```sh
mflux-train --config train.json
```

Iris works on pixels, so a training image is used as it is, with no VAE. The noise levels are an even grid of 1000 levels from pure noise, as in the reference trainer, shifted by resolution as the [reference trainer](https://github.com/speridlabs/iris-3b) shifts them: 2 at 256×256, 3 at 512×512 and 4 at 1024×1024. Each image gets the shift of its own size. Leave `timestep_type` unset to draw uniformly from that grid. The reference trainer draws from a logit-normal distribution (`"timestep_type": "sigmoid"`), which rarely reaches the highest noise levels. In our style tests it learned the colors less reliably (numbers below); on a character dataset the two came out the same.

A config for the 24 transformer blocks with rank 16, the layers ai-toolkit trains. All 311 linear layers can take a target:

```json
{
  "model": "iris-3b-base",
  "data": "/path/to/images",
  "seed": 42,
  "steps": 30,
  "guidance": 3.0,
  "max_resolution": 512,
  "gradient_checkpointing": true,
  "cache_limit_gb": 2,
  "training_loop": { "num_epochs": 25, "batch_size": 1 },
  "optimizer": { "name": "AdamW", "learning_rate": 1e-4 },
  "checkpoint": { "output_path": "output", "save_frequency": 300 },
  "lora_layers": { "targets": [
    { "module_path": "blocks.{block}.attn.q_proj_x", "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn.q_proj_y", "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn.k_proj_x", "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn.k_proj_y", "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn.v_proj_x", "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn.v_proj_y", "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn.proj_x",   "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn.proj_y",   "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn_gate_x",   "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn_gate_y",   "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp_x.w1",      "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp_x.w2",      "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp_x.w3",      "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp_y.w1",      "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp_y.w2",      "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp_y.w3",      "blocks": { "start": 0, "end": 8 }, "rank": 16 },
    { "module_path": "blocks.{block}.q_proj",        "blocks": { "start": 8, "end": 24 }, "rank": 16 },
    { "module_path": "blocks.{block}.k_proj",        "blocks": { "start": 8, "end": 24 }, "rank": 16 },
    { "module_path": "blocks.{block}.v_proj",        "blocks": { "start": 8, "end": 24 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn_gate",     "blocks": { "start": 8, "end": 24 }, "rank": 16 },
    { "module_path": "blocks.{block}.attn_proj",     "blocks": { "start": 8, "end": 24 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp.w1",        "blocks": { "start": 8, "end": 24 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp.w2",        "blocks": { "start": 8, "end": 24 }, "rank": 16 },
    { "module_path": "blocks.{block}.mlp.w3",        "blocks": { "start": 8, "end": 24 }, "rank": 16 }
  ] }
}
```

`steps` and `guidance` are for the preview images; training draws from its 1000 levels whatever `steps` is. Blocks 0 to 7 are dual-stream (`_x` for the image, `_y` for the text) and 8 to 23 single-stream. Once the captions are encoded, mflux encodes the preview prompts and lets the 8.9 GB Qwen3-VL text encoder go. The LoRA is saved with ai-toolkit's keys and shapes (`diffusion_model.*`, PEFT `lora_A`/`lora_B`), and loads with `--lora`.

Against ai-toolkit on an M5 Max, with the same 12 images and the config above (300 steps at 512×512, rank 16, gradient checkpointing on both), 3 runs each:

| | per step | peak memory | style score |
| --- | --- | --- | --- |
| ai-toolkit | 1.23 to 1.37 s | 17.1 GB | 0.88, 0.86, 0.87 |
| mflux | 1.03 to 1.24 s | 10.8 GB¹ | 0.86, 0.88, 0.86 |
| mflux, `"timestep_type": "sigmoid"` | 1.03 to 1.15 s | 18.8 GB² | 0.67, 0.36, 0.77 |

The style score is the share of pixels in the dataset's blueprint blue on 3 prompts outside the training images, each LoRA generated in mflux. Runs at the same time of day took 10 to 17% less time per step in mflux. ¹ Run 1 repeated with the text encoder released, same score; the three runs kept it and peaked at 18.8 GB. ² With the text encoder kept. These runs drew from a 30-level grid; run 1 repeated on the 1000 levels training uses now scored the same, 0.86. `"quantize": 8` trains over a quantized base: a short run with previews on peaked at about 13 GB.

Without `"low_ram": true` the runner keeps every caption's encoding in memory, about 18 MB per image: nothing for these 12 images, about 9 GB for 500. Set it for a large dataset.

For speed over memory, set `"gradient_checkpointing": false`: a step took 0.78 s instead of 1.05 s, and the peak went from 10.7 to 21 GB. Releasing the text encoder costs no time, since training never runs it again once the captions are encoded.

## Quantization
`-q 4` and `-q 8` quantize the model when it loads, the transformer and the text encoder. Small layers (the pixel-stage embedder, MLPs and head, the 12-to-1 layer pooling) and the feed-forward output projections, whose input width of 6826 is not a multiple of the quantization group size, stay in bf16. At `-q 4` and below the per-pixel adaLN projections of the pixel stage (2560 to 16384, they set individual pixel values) stay in bf16 too: quantized to 4 bits they print a faint screen-door grain over flat areas and fur. Measured on an M4 Pro (64 GB), Turbo, 1024×1024, seed 42. The transformer and the text encoder are both in memory while the weights load and the prompt is encoded; mflux then releases the text encoder, which is why the peak while generating is lower. The PSNR compares each image with the unquantized one: the composition is the same, the fine detail differs, and `-q 4` is visibly softer.

| | On disk (`mflux-save`) | Peak while loading + encoding | Peak while generating | Time (4 steps) | PSNR vs bf16 |
|---|---|---|---|---|---|
| bf16 | – | 14.5 GB | 7.6 GB | 17.2 s | – |
| `-q 8` | 7.5 GB (transformer 3.5 GB, text encoder 4.0 GB) | 8.8 GB | 5.3 GB | 18.4 s | 34.6 dB |
| `-q 4` | 4.7 GB (transformer 2.6 GB, text encoder 2.1 GB) | 5.9 GB | 4.4 GB | 18.2 s | 20.7 dB |

MLX reads weights lazily, so the first image also pays for loading them from disk. Quantization saves memory, not time. Save a quantized model once and load it from disk:

```sh
mflux-save --model iris-3b --quantize 8 --path ~/models/iris-3b-q8
mflux-generate-iris --model ~/models/iris-3b-q8 --prompt "..."
```

The saved directory holds the transformer, the text encoder and the tokenizer, so it loads without the Hugging Face repositories. An image from the saved model is identical to the image of the same `-q` flag loaded from Hugging Face.

> [!IMPORTANT]
> A saved model is identified by its directory name, as for FLUX.2 and FLUX 3: the bare name is Turbo (`iris-3b-q8`, `iris-q8`, `my-iris`) and a **base** model needs `iris-3b-base` or `iris-base` in the name, or `--base-model iris-3b-base`. For example, save the base model to `~/models/iris-3b-base-q8`.

## Speed and size
- Turbo, 1024×1024, bf16: about 4.4 s per step on an M4 Pro, 18 s per image after the model is loaded.
- Base, 1024×1024, bf16: 8.6 s per step (two passes), so about 14 minutes for the default 100 steps.
- `--low-ram` reduces the memory of the loop little for this model, since there is no VAE to tile. It still releases the text encoder and the transformer when they are no longer needed.

> [!WARNING]
> mflux downloads the transformer (5.6 GB for Turbo, 11 GB for the base model; the repository holds both) and the Qwen3-VL-4B-Instruct checkpoint (8.2 GB, of which only the language tower is used).

## Notes
- The prompt is wrapped in the chat template the model was trained with and limited to 300 tokens in total. The template takes 5 of them, so a caption is cut after 295 tokens.
- Sizes must be multiples of 16. Turbo was trained at about one megapixel: much smaller sizes show 16×16 patch artifacts, in the reference implementation too.
- `mflux-generate-iris` has no `--scheduler`, `--compute-precision` or `--vae-tiling` options: the model has no VAE and the samplers are part of the checkpoints. The transformer runs the attention queries and keys in float32; with bf16 queries and keys, single patches of flat regions came out visibly off.
- Both entries share one Hugging Face repository. `--model iris-3b` (default, aliases `iris-3b-turbo`, `iris-turbo`, `iris`) selects Turbo, `--model iris-3b-base` (alias `iris-base`) the base model.
- Not ported from the reference implementation: depth estimation and the upscaler fine-tunes.
- Slow golden tests are not part of this port; `tests/` holds fast tests for the weight mapping, quantization, prompt encoding, samplers, CLI and saving.

## License and attribution
Iris-3B is released by [Speridlabs](https://github.com/speridlabs/iris-3b) under the Apache License 2.0, and Qwen3-VL-4B-Instruct (the text encoder) by Alibaba Cloud under the Apache License 2.0.
