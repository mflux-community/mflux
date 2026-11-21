# MOSS-Image

This directory contains the MFLUX MLX port of **MOSS-Image-SFT**
([`OpenMOSS-Team/MOSS-Image-SFT`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)) and
**MOSS-Image-Base** ([`OpenMOSS-Team/MOSS-Image-Base`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)).

MOSS-Image-SFT is a text-to-image model from OpenMOSS (Apache 2.0). It is a 4.9B
single-stream diffusion transformer with a Qwen3.5-9B text encoder and the FLUX.2 VAE with
the small decoder. It accepts natural-language prompts, tag sequences and JSON layouts, and it
renders English and Chinese text.

MOSS-Image-Base is the base model of the family. It has the same architecture, text encoder, VAE
and sampling settings as SFT. Only the transformer weights are different. To run it, add
`--model moss-image-base`.

The port runs all three parts in MLX. The text encoder is an MLX port of the hybrid Qwen3.5
decoder (Gated DeltaNet and gated attention layers), so the model does not need PyTorch at
run time.

## Example

```sh
mflux-generate-moss-image \
  --prompt 'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in English: "Buy milk and eggs!"' \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  -q 8
```

The defaults follow the model card: 50 steps, guidance 3 and an empty negative prompt. The
negative branch of CFG skips transformer layer 22. Each step runs the transformer two times.
`--uncond-skip-layers` sets other layers, and the flag with no values skips no layer. The image
metadata records the skipped layers, so `--config-from-metadata` replays them.

- Width and height must be multiples of 16.
- `--steps` accepts 1 to 218. Above 218, two steps snap to the same point of the training
  schedule, so the command stops with an error.
- `--scheduler` is ignored. The model samples on the schedule it was trained with.
- `--step-cache-ratio 0.25` skips the transformer on about a quarter of the steps. The
  timestep embedding of the model selects the steps.

Quantization (`-q 3|4|5|6|8`) applies to the transformer and the text encoder. To keep a
quantized copy, save it once and load the copy with `--model`:

```sh
mflux-save --model moss-image -q 4 --path ~/models/moss-image-q4
mflux-generate-moss-image --model ~/models/moss-image-q4 --prompt "a lighthouse at dusk"
```

On an M-series Mac with 36 GB, a 1024x1024 image with the 4-bit copy uses about 10.5 GB of
MLX memory and takes about 9.7 seconds for each step. The 8-bit model at 512x512 takes about
2.2 seconds for each step. With `--low-ram`, a 1024x1024 image peaks at about 7.6 GB at 4-bit
(the VAE decode) and about 9.3 GB at 8-bit (the text encoder), measured on an M4 Pro.

<details>
<summary>Python API</summary>

```python
from mflux.models.moss_image import MOSSImage

model = MOSSImage(quantize=8)
image = model.generate_image(
    seed=42,
    prompt="A majestic fluffy cat sitting on a wooden bench, golden hour lighting, photo",
    width=1024,
    height=1024,
    num_inference_steps=50,
    guidance=3.0,
)
image.save(path="moss_cat.png")
```

`generate_image` also takes `negative_prompt`, `uncond_skip_layers` (default `(22,)`, an empty
tuple turns the layer skip off) and `step_cache_ratio`.

`MOSSImage` takes `lora_paths`, `lora_scales` and `bake_lora` (default `False`). For the base
model, give `model_config=ModelConfig.moss_image_base()`
(`from mflux.models.common.config import ModelConfig`).

You can also call the steps of `mflux-generate-moss-image` from Python. `MOSSImageCommand` in
`mflux.models.moss_image.cli.moss_image_generate` has `validate(args)`, `load(args)` and
`generate(model, args, seed, prompt)`.

</details>

## LoRA

`--lora PATH [SCALE]` applies MOSS-Image LoRAs. mflux reads the kohya format
(`diffusion_model.layers.N.proj.q.lora_down.weight`, with an `alpha`) and the PEFT format
(`layers.N.proj.q.lora_A`). These LoRAs change the six projections of each transformer layer:
`q`, `k`, `v`, `o`, `ffn_up_gate` and `ffn_down`.

The step-distilled LoRAs from the community let the SFT model make an image in 8 or 4 steps
without CFG:

```sh
mflux-generate-moss-image \
  --lora lodestones/MOSS-Image-SFT-8-steps-dmd:lora.safetensors \
  --steps 8 \
  --guidance 1 \
  --prompt "a red fox sitting in fresh snow at dawn, a wooden sign behind it that says MOSS"
```

[`darask0/MOSS-Image_4-step_adapter_lora`](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora)
works the same way with `--steps 4`. Both repositories also have a `time_embedding.safetensors`
file and a `dataspace.safetensors` file for their own sampler. mflux does not use these files.
With the MOSS-Image schedule and CFG 1, the LoRA alone gives clean images. Both repositories
name the LoRA `lora.safetensors`, and the mflux LoRA cache keeps one link for each file name.
Thus after you use one of them, `repo:lora.safetensors` of the other gives the same file. Give
a local path for the second one.

On this model, a LoRA stays a separate layer by default (`--bake-lora` is off). Most of the
update of the step-distilled LoRAs is smaller than half a bfloat16 step of the weights. A merge
into the bfloat16 weights keeps 44% (8-step LoRA) and 49% (4-step LoRA) of the update. Then some
images have soft detail, a malformed animal or wrong text. The separate layers apply all of the
update. They use about 1.2 GB more memory for a rank-128 LoRA. In single runs on an M5 Max at
768x768, they took 5% to 17% more time for each step than the merged weights.

`--bake-lora` with `-q 8` or `-q 4` adds the LoRA to the weights in float32 before it quantizes
them, and keeps all of the update. To keep a quantized copy with a step-distilled LoRA in it,
save it with `-q 8`:

```sh
mflux-save --model moss-image -q 8 --lora lodestones/MOSS-Image-SFT-8-steps-dmd:lora.safetensors --path ~/models/moss-image-8-step-q8
mflux-generate-moss-image --model ~/models/moss-image-8-step-q8 --steps 8 --guidance 1 --prompt "a lighthouse at dusk"
```

A copy that you save without `-q` keeps about half of the update of these LoRAs, for the reason
above. On a copy that is already quantized, `--bake-lora` can only merge the LoRA into the
quantized weights, and that keeps about 20% of the update. Keep the default there.

[`chinoll/Moss-Image-Edit`](https://huggingface.co/chinoll/Moss-Image-Edit) is an image-editing
LoRA in the PEFT format. mflux loads it, but this port does not take reference images, so it
cannot edit with it.

## Weights

The model loads the official repository `OpenMOSS-Team/MOSS-Image-SFT` or
`OpenMOSS-Team/MOSS-Image-Base` (about 27 GB of bf16 weights each) or a copy that `mflux-save`
wrote. The loader stops with an error when a checkpoint
does not have all the weights of the model. The MLX conversions in the `vanch007/MOSS-Image-SFT-mlx-*`
repositories use other weight names and do not include the text encoder, so this port does not load them.

## Streaming from the SSD (`--ssd-stream`)

The model is 26 GB in bf16, more than the RAM of many Macs. `--ssd-stream` runs the 32 text-encoder layers and the 42 transformer blocks one at a time from the SSD. See [SSD streaming](../common/README.md#ssd-streaming---ssd-stream) for how it works and how to check that a run does not write to the SSD.

Get the files. The model must be a `mflux-community` repo (or an `mflux-save` folder), because the streamer reads the saved layers by name:

```sh
hf download mflux-community/moss-image-sft-mflux-bf16   # 26.18 GB: text encoder 15.9, transformer 10.2, VAE 0.1
hf download lvladikov/MOSS-SFT-Distill-4step-LoRA _archive/checkpoints/moss_sft_4step_rank_64_lora_chk00015000.safetensors   # 594.6 MB, checkpoint 15000
```

Run the 4-step LoRA with it:

```sh
mflux-generate-moss-image \
  --model mflux-community/moss-image-sft-mflux-bf16 --base-model moss-image-sft \
  --lora lvladikov/MOSS-SFT-Distill-4step-LoRA:_archive/checkpoints/moss_sft_4step_rank_64_lora_chk00015000.safetensors \
  --steps 4 --guidance 1 --width 1024 --height 1024 --seed 42 \
  --ssd-stream --mlx-cache-limit-gb 0.256 \
  --prompt "a fisherman mending a net on a foggy pier, a lighthouse behind him"
```

- A LoRA loads as usual and stays in memory (594.6 MB for this one).
- With `--stream-lora` as well, the one `--lora` file stays on the SSD too: each block's part of it is read just before the block runs, so the whole LoRA is not resident. It reads the same MOSS-Image LoRA formats as the loader (kohya and PEFT names, with or without a prefix, alpha from the tensors, the PEFT metadata or the file metadata) and the projections that the file trains. A test checks that its output equals the loader's for each alpha source. It stops with an error, before it reads any weight, for `--stream-lora` without `--ssd-stream`, for no `--lora` or more than one, and for `--bake-lora`. It also stops for a DoRA file, for a file with only one of the two matrices of a projection, and for a file that names no projection of the model, as the loader does. Without `--stream-lora`, a LoRA loads as usual and stays in memory.
- Only checkpoint 15000 was run. The author's newer checkpoints have the same size and layout and should work, but they have not been tested.
- With the original `OpenMOSS-Team` repo the text encoder is not streamed, because its layer names differ.
- `--ssd-stream` also decodes the image in row strips, as the FLUX.2 models do, because the full-image decode used more memory than the rest of the run. At 512x512 (one run, mflux 0.24.0) the strip image differed from the full-decode image by 0.40 of 255 in mean absolute pixel value. With the strips, `--vae-tiling` and `--vae-tile-size` have no effect, and the command prints a warning when you pass them. Without `--ssd-stream` the decode is unchanged.
- `--ssd-stream` stops with an error, before it reads any weight, in these cases, because each one would otherwise give a wrong image without a message:
  - with `-q`: the streamer installs the saved weights over the layers, and layers that `-q` has quantized need packed weights. To stream a quantized model, use one that is already quantized, such as a pre-quantized repo or a copy saved with `mflux-save -q N`, and do not add `-q`. Streaming a quantized model has not been run.
  - with `--bake-lora` and a `--lora`: the streamer would install the saved weights over the baked ones and drop the LoRA. Run the LoRA without `--bake-lora` (the default).
- A model that was loaded with `ssd_stream=True` cannot be saved: `save_model` raises an error, because the idle layers are placeholders.
- A pass that skips transformer layers (`--uncond-skip-layers`) reads the block after each running block ahead, and then jumps over it. That block is dropped at the next call, so the skipped blocks do not pile up in memory, but the block that runs after the jump is read when it is needed, without the overlap.
- Measured on an Apple M1 with 8 GiB and macOS 14.6.1, bf16, 4 steps, CFG 1, 4-step LoRA at 1.0, checkpoint 15000, seed 42, one run each.

  With mflux 0.24.0 and a separate script that did the same streaming and also read the LoRA from the SSD one block at a time:

  | Size | Time | SSD writes | Swap-outs | Peak footprint |
  |---|---|---|---|---|
  | 1024x1024 | 149 s | 14 MB | 0 | 3.0 GB |
  | 1824x1024 | 305 s | 21 MB | 0 | 2.96 GB |

  With this code (mflux 0.24.0 plus the commits after it, MLX 0.32.0, the LoRA kept in memory):

  | Size | Time | SSD writes | Swap-outs | Peak footprint |
  |---|---|---|---|---|
  | 1024x1024 | 151.6 s | 42 MB (about 29 MB above the idle level) | 0 | 3.36 GB |

## Validation

We compared the port with the reference pipeline (`pipeline_moss_image.py` in the model
repository) on the same weights:

- The transformer matches the reference to 6e-7 in float32 on random weights.
- The Qwen3.5 text model matches `transformers` to 5e-5 in float32 on random weights.
- With the real weights in bf16, the token ids are equal and the text embeddings have a
  cosine similarity of 0.999 or more.
- With the same initial noise and text embeddings (512x512, 20 steps, CFG 3), the MLX image
  has a PSNR of 34.9 dB against the PyTorch image in bf16 and 36.6 dB in 8-bit.
