# DreamLite
This directory contains the MFLUX MLX port of **DreamLite**
([`ByteVisionLab/DreamLite`](https://github.com/ByteVisionLab/DreamLite)).

DreamLite is a compact model from ByteDance for on-device use. One 0.39B UNet does both **text-to-image generation**
and **instruction-based image editing**. MFLUX supports the two released checkpoints:

| Model | `--model` | Steps | Guidance |
| --- | --- | --- | --- |
| [DreamLite-base](https://huggingface.co/carlofkl/DreamLite-base) | `dreamlite` (default) | 28 | 3.5, with a negative prompt |
| [DreamLite-mobile](https://huggingface.co/carlofkl/DreamLite-mobile) | `dreamlite-mobile` | 4 | none (step-distilled) |

Quantization (`-q 4`, `-q 8`), `mflux-save`, `--compute-precision float16`, `--low-ram`, step-by-step previews and
metadata export work as for other models.

![DreamLite Example](../../assets/dreamlite_example.jpg)

*Left to right: DreamLite-base text-to-image (28 steps), DreamLite-mobile text-to-image (4 steps), and two edits of the
second image: to an oil painting by base, and to a forest background by mobile. All from the examples below, seed 42.*

> [!WARNING]
> The DreamLite weights are for non-commercial research use only, and their licence does not allow redistribution.
> Read the [weights licence](https://github.com/ByteVisionLab/DreamLite/blob/main/WEIGHTS_LICENSE) before you use
> them. The first run downloads about 5 GB for each of the two models.

## Example
```sh
mflux-generate-dreamlite \
  --prompt "A close-up of a fire spitting dragon, cinematic shot." \
  --seed 42
```

`--steps` defaults to 28 and `--guidance` to 3.5, the settings of the reference scripts. With guidance above 1, every
step runs the UNet two times, once for the prompt and once for the negative prompt (`--negative-prompt`, empty by
default).

DreamLite-mobile makes an image in 4 steps without guidance:

```sh
mflux-generate-dreamlite \
  --model dreamlite-mobile \
  --prompt "A portrait of a young woman with flowers." \
  --seed 42 \
  --output portrait.png
```

- Width and height must be multiples of 32. The default size is 1024×1024, the size the model was trained for.
- DreamLite-mobile runs without guidance, as in the reference pipeline. `--guidance` and `--negative-prompt` have no
  effect on it, and mflux warns when you give them.
- `--scheduler` and the VAE tiling flags are ignored.

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.dreamlite import DreamLite

model = DreamLite()
image = model.generate_image(
    seed=42,
    prompt="A close-up of a fire spitting dragon, cinematic shot.",
    num_inference_steps=28,
    guidance=3.5,
)
image.save(path="dragon.png")

mobile = DreamLite(model_config=ModelConfig.dreamlite_mobile())
image = mobile.generate_image(seed=42, prompt="A portrait of a young woman with flowers.")
image.save(path="portrait.png")
```

You can also call the steps of `mflux-generate-dreamlite` from Python. `DreamLiteCommand` in
`mflux.models.dreamlite.cli.dreamlite_generate` has `validate(args)`, `load(args)` and
`generate(model, args, seed, prompt)`.

</details>

## Editing
`mflux-generate-dreamlite-edit` changes an image as the prompt tells it to. The prompt is the instruction. These
commands edit the portrait from the example above:

```sh
mflux-generate-dreamlite-edit \
  --image-paths portrait.png \
  --prompt "Transfer this image to oil-painting style." \
  --seed 42
```

```sh
mflux-generate-dreamlite-edit \
  --model dreamlite-mobile \
  --image-paths portrait.png \
  --prompt "Change the background to a dense forest." \
  --seed 42
```

- The command takes one image.
- As in the reference pipeline, the edit runs on a 1024×1024 copy of the image, whatever its aspect ratio. mflux saves
  the result at the size of the source image. `--width` and `--height` set another size; `--width 2x` scales the
  source size. Images that are far from square lose quality, because the model sees them squeezed.
- On DreamLite-base, `--guidance` (default 3.5) sets how strongly the result follows the instruction, and
  `--image-guidance` (default 1.0) how strongly it keeps to the source image. Each step runs the UNet two times. An
  image guidance other than 1.0 adds a third pass without the source image.
- DreamLite-mobile edits in 4 steps, one UNet pass each, and uses neither guidance value.

<details>
<summary>Python API</summary>

```python
from mflux.models.dreamlite import DreamLiteEdit

model = DreamLiteEdit()
image = model.generate_image(
    seed=42,
    prompt="Transfer this image to oil-painting style.",
    image_path="portrait.png",
)
image.save(path="portrait_oil.png")
```

`DreamLiteEditCommand` in `mflux.models.dreamlite.cli.dreamlite_edit_generate` has the same `validate`, `load` and
`generate` steps.

</details>

## How it works
- **In-context spatial concatenation.** The UNet always sees two images side by side in latent space: the image it
  denoises on the left, and a context image on the right. For an edit the context is the source image. For
  text-to-image it is empty. Only the left half of the UNet output is used.
- **Text encoder.** The prompt goes through Qwen3-VL-2B. For an edit, Qwen3-VL also sees a small copy of the source
  image (512 px on base, 256 px on mobile) beside the instruction.
- **UNet.** A pruned SDXL-style UNet with depthwise-separable convolutions and multi-query attention. Only the lowest
  resolution has self-attention.
- **VAE.** TAESDXL, a tiny autoencoder with 2.4M weights.

## Quantization and memory
Quantization (`-q 4`, `-q 8`) applies to the linear layers of the UNet and the text encoder. The text encoder is most of
the model, so it is also most of what quantization saves. To keep a quantized copy, save it once and load the copy with
`--model`:

```sh
mflux-save --model dreamlite-mobile -q 8 --path ~/models/dreamlite-mobile-q8
mflux-generate-dreamlite --model ~/models/dreamlite-mobile-q8 --base-model dreamlite-mobile --prompt "a lighthouse at dusk"
```

Measured on an M4 with 24 GB, for a 1024×1024 image. The memory is the peak of the whole process:

| | Size on disk | Peak memory | Time, with the load |
| --- | --- | --- | --- |
| DreamLite-mobile, as downloaded | 4.8 GB | 4.4 GB | 9 s |
| DreamLite-mobile, `-q 8` | 2.7 GB | 4.4 GB | 9 s |
| DreamLite-mobile, `-q 4` | 1.5 GB | 4.1 GB | 9 s |
| DreamLite-mobile, `-q 4 --compute-precision float16` | 1.5 GB | 3.2 GB | 7.5 s |
| DreamLite-base, as downloaded | 4.8 GB | 4.6 GB | 75 s |

A UNet pass takes about 1.2 seconds at every quantization level. DreamLite-mobile runs 4 of them, DreamLite-base 56
(two in each of its 28 steps). An edit needs more memory, because Qwen3-VL also reads the image: 5.6 GB on mobile and
6.5 GB on base, in 11 s and 85 s. A copy that `mflux-save` writes without `-q` is 5.4 GB, because it stores the UNet
in float32.

Where the memory goes:

- The text encoder is the largest part: 3.4 GB as downloaded, about 2 GB at `-q 8` and about 1 GB at `-q 4`. mflux
  frees it before the denoising loop when the command makes one image.
- A UNet pass needs about 3 GB (1.5 GB of float32 weights and its activations), which is the peak once the text
  encoder is quantized. This is why `-q 8` saves little memory, and `--compute-precision float16` saves more.
- Both commands limit the buffer cache of MLX to 1 GB. Without a limit MLX keeps every buffer that a UNet pass frees,
  and the process holds 9 to 10 GB for a step that is 2 to 6% faster. `--mlx-cache-limit-gb` sets another limit. The
  `Peak MLX memory` line that the command prints does not include this cache. In Python, call
  `mx.set_cache_limit(1000**3)` before you generate.

`-q 8` gives nearly the same image as the unquantized model (PSNR 28 dB). `-q 4` gives a different image of the same
quality: the UNet reads the text encoder's last hidden states directly, and they change with 4-bit weights. The
convolutions, the normalization layers and the timestep and size embedders of the UNet are not quantized.

## Compute precision
`--compute-precision float16` runs the UNet in float16. The text encoder and the VAE keep their own precision. A
UNet pass then takes about 1.0 second in place of 1.2:

```sh
mflux-generate-dreamlite \
  --model dreamlite-mobile \
  --prompt "A portrait of a young woman with flowers." \
  --seed 42 \
  --compute-precision float16
```

| | Default (float32) | `--compute-precision float16` |
| --- | --- | --- |
| DreamLite-mobile, time with the load | 9 s | 8 s |
| DreamLite-base, time with the load | 73 s | 57 s |
| PSNR against the reference image, text-to-image | 67 to 74 dB | 43 to 47 dB |
| PSNR against the reference image, edit | 65 to 80 dB | 34 to 50 dB |

The image changes slightly for the same seed. The difference grows with the number of steps, so it is largest in a
28-step edit on DreamLite-base. The option works together with `-q`. The run records it in the metadata, and
`--config-from-conf` replays it. `mflux-save` does not take the option: a saved model holds the weights, not the
compute precision.

In Python, pass `compute_precision=mx.float16` to `DreamLite` or `DreamLiteEdit`.

## Weights
The models load the `main` branch of [`carlofkl/DreamLite-base`](https://huggingface.co/carlofkl/DreamLite-base) and
[`carlofkl/DreamLite-mobile`](https://huggingface.co/carlofkl/DreamLite-mobile), each pinned to one commit, or a copy
that `mflux-save` wrote. mflux downloads the `unet`, `vae`, `text_encoder` and `processor` folders only.

The checkpoints store bfloat16 weights. mflux runs the UNet and the VAE in float32, which is faster on Apple silicon
than bfloat16 for this convolutional network and keeps the image at the reference's. The text encoder keeps its
bfloat16 weights and computes its activations in float32.

## Not supported
- LoRA. The reference repository has LoRA training scripts; mflux does not load DreamLite LoRAs.
- Image-to-image with `--image-path` and an image strength. Use `mflux-generate-dreamlite-edit` to change an image.
- Training with `mflux-train`.
- The aspect-ratio buckets of the reference pipeline for edits (`bucket=54` and `bucket=765`). Its default, the
  1024×1024 edit, is what mflux runs.

## Validation
We compared the port with the reference pipeline (`ByteVisionLab/DreamLite` in PyTorch, float32) on the same weights
and the same initial noise:

- The text encoder output, the VAE encoder and decoder output and single UNet calls match the reference: the largest
  difference in a UNet output is under 1e-3, with values of standard deviation 0.8.
- The final images match the reference in every mode:

| Mode | PSNR against the reference image |
| --- | --- |
| DreamLite-mobile, text-to-image, 1024×1024, 4 steps | 67.3 dB |
| DreamLite-mobile, text-to-image, 1280×768, 4 steps | 74.3 dB |
| DreamLite-base, text-to-image, 1024×1024, 28 steps, guidance 3.5 | 71.8 dB |
| DreamLite-mobile, edit, 4 steps | 79.7 dB |
| DreamLite-base, edit, 28 steps, guidance 3.5 | 65.3 dB |
| DreamLite-base, edit, 12 steps, guidance 3.5, image guidance 1.5, with a negative prompt | 67.0 dB |

Speed against the reference on the same M4, for a 1024×1024 image once the model is loaded:

| | DreamLite-mobile | DreamLite-base |
| --- | --- | --- |
| Reference, PyTorch on MPS, float32 | 6.4 s | 87.7 s |
| Reference, PyTorch on MPS, bfloat16 | 5.1 s | 62.9 s |
| mflux | 5.1 s | 69.0 s |
| mflux, `--compute-precision float16` | 4.0 s | 53.7 s |

MLX and PyTorch make different noise from the same seed, so an mflux image and a reference image from the same seed
are different images of the same kind.
