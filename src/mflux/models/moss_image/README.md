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

## Validation

We compared the port with the reference pipeline (`pipeline_moss_image.py` in the model
repository) on the same weights:

- The transformer matches the reference to 6e-7 in float32 on random weights.
- The Qwen3.5 text model matches `transformers` to 5e-5 in float32 on random weights.
- With the real weights in bf16, the token ids are equal and the text embeddings have a
  cosine similarity of 0.999 or more.
- With the same initial noise and text embeddings (512x512, 20 steps, CFG 3), the MLX image
  has a PSNR of 34.9 dB against the PyTorch image in bf16 and 36.6 dB in 8-bit.
