# SenseNova-U1.5
This directory contains the MFLUX MLX port of **SenseNova-U1.5-8B-MoT**
([`sensenova/SenseNova-U1.5-8B-MoT`](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)).

SenseNova-U1.5 is a unified multimodal model from SenseTime (Apache 2.0). It is a Mixture-of-Transformers on
Qwen3-8B: every layer has two sets of weights, one for the understanding stream and one for the generation stream.
There is no VAE and no latent space. The generation stream denoises RGB pixels, patches of 32×32 pixels become one token, and a
convolutional head turns the tokens back into pixels.

For text-to-image the two streams run one after the other. The prompt goes through the understanding stream once and
leaves a key/value prefix for each layer. The generation stream then denoises the image and attends to that prefix and to
its own tokens, without a mask. The prompt encoder runs again only when the prompt changes, and mflux frees it before
the denoising loop under `--low-ram`.

This port covers **text-to-image only**. Image editing, think mode and visual question answering are not ported.

## Example
```sh
mflux-generate-sensenova \
  --prompt 'A red fox sleeping in fresh snow at golden hour, a wooden sign that reads "MLX".' \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  -q 8
```

The defaults follow the model card: 50 steps and guidance 4.0. With guidance above 1, every step runs the generation
stream two times, once for the prompt and once for an empty query. SenseNova-U1.5 has no negative prompt, so
`--negative-prompt` is an error. `--scheduler` and the VAE flags are ignored.

- Width and height must be multiples of 32. The default size is 1024×1024.
- The model was trained on these resolutions: 2048×2048, 2720×1536, 1536×2720, 2496×1664, 1664×2496, 2368×1760,
  1760×2368, 1440×2880, 2880×1440, 1152×3456 and 3456×1152. 1024×1024 also works well and is about 4 times faster.
  mflux does not warn about other sizes.

<details>
<summary>Python API</summary>

```python
from mflux.models.sensenova import SenseNova

model = SenseNova(quantize=8)
image = model.generate_image(
    seed=42,
    prompt='A red fox sleeping in fresh snow at golden hour, a wooden sign that reads "MLX".',
    width=1024,
    height=1024,
    num_inference_steps=50,
    guidance=4.0,
)
image.save(path="sensenova_fox.png")
```

`SenseNova` also takes `lora_paths`, `lora_scales` and `bake_lora` (default `False`). `generate_image` raises when it
gets a `negative_prompt`.

You can also call the steps of `mflux-generate-sensenova` from Python. `SenseNovaCommand` in
`mflux.models.sensenova.cli.sensenova_generate` has `validate(args)`, `load(args)` and `generate(model, args, seed, prompt)`.

</details>

## 8-step LoRA
SenseTime publishes a step-distilled LoRA for this model in
[`sensenova/SenseNova-U1.5-8B-MoT-LoRAs`](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs). With it, the
model makes an image in 8 steps without CFG, which takes about a tenth of the time of the 50-step run:

```sh
mflux-generate-sensenova \
  --lora sensenova/SenseNova-U1.5-8B-MoT-LoRAs:SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors \
  --steps 8 \
  --guidance 1 \
  --prompt 'A red fox sleeping in fresh snow at golden hour, a wooden sign that reads "MLX".' \
  -q 8
```

mflux does not set `--steps 8 --guidance 1` for you: the LoRA file has no step metadata, so the command keeps its 50
steps and guidance 4.0 unless you give both flags. mflux reads the kohya format (`lora_down`, `lora_up` and a per-layer
`alpha`, with or without a `diffusion_model.` prefix) and the PEFT format. The LoRA changes the seven linear layers of the
generation stream in each of the 42 layers: `q`, `k`, `v`, `o`, `gate`, `up` and `down`.

On this model, a LoRA stays a separate layer on top of the weights by default (`--bake-lora` is off), so all of the
update applies. `--bake-lora` merges it into the weights instead, which saves about 1 GB and 10% of the time per step:

- With `-q`, the merge happens in float32 before the weights are quantized. This gives the quantized image closest to
  the unmerged bfloat16 one (PSNR 34 dB, against 29 dB for `-q 8` with the LoRA left separate).
- Without `-q`, the merge rounds into the bfloat16 weights and keeps about 82% of the update. That is what the upstream
  PyTorch pipeline does (36 dB to its images), so use it to reproduce upstream bfloat16 images.

`mflux-save` accepts the LoRA, and the saved copy has it merged in. Quantize it with `-q`, so that the merge happens in
float32. A saved 8-step model still needs `--steps 8 --guidance 1`:

```sh
mflux-save --model sensenova-u1.5 -q 4 --lora sensenova/SenseNova-U1.5-8B-MoT-LoRAs:SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors --path ~/models/sensenova-8-step-q4
mflux-generate-sensenova --model ~/models/sensenova-8-step-q4 --steps 8 --guidance 1 --prompt "a lighthouse at dusk"
```

## Quantization and memory
Quantization (`-q 3|4|5|6|8`) applies to the transformer and the text encoder. The checkpoint is 35 GB in bfloat16,
so use `-q 4` or `-q 8` on a Mac with 16 to 24 GB. To keep a quantized copy, save it once and load the copy with `--model`:

```sh
mflux-save --model sensenova-u1.5 -q 4 --path ~/models/sensenova-q4
mflux-generate-sensenova --model ~/models/sensenova-q4 --prompt "a lighthouse at dusk"
```

Measured on an M4 Pro with `--low-ram`, the 8-step LoRA and a 1024×1024 image:

| Precision | Peak memory, including the load | Resident while denoising |
| --- | --- | --- |
| `-q 4` | 11.5 GB | 5.5 GB |
| `-q 8` | 19.6 GB | 9.6 GB |
| bf16 | 34.9 GB | 17.2 GB |

The peak is the load, when both streams are in memory; the peak that `mflux-generate-sensenova` prints is measured
after the load, so it is lower. A step takes about 3.1 seconds at every precision, so an image with
the LoRA takes about 25 seconds. The base model with 50 steps and CFG takes about 5.1 seconds for each step.

## Weights
The model loads the official repository `sensenova/SenseNova-U1.5-8B-MoT` (35 GB of bfloat16 weights, pinned to one
commit) or a copy that `mflux-save` wrote. The loader stops with an error when a checkpoint does not have all the
weights of the model. The text-to-image path does not use the language-model head or the vision tower of the
understanding stream, so mflux does not load them.

## Validation
We compared the port with the reference pipeline (`OpenSenseNova/SenseNova-U1`) on the same weights:

- With 4 layers in float32, every stage of the denoising step (patch embedding, timestep embeddings, hidden states, head
  output and velocity) and the prompt key/value prefix match the reference to 7e-6 relative error. This holds for
  square and rectangular sizes, and with the LoRA, merged and unmerged.
- With the real weights in bfloat16 (1024×1024, 50 steps, CFG 4, same initial noise), the MLX image has a PSNR of 30.8 dB
  against the PyTorch image. With the 8-step LoRA merged as upstream does it, the PSNR is 36.4 dB.
