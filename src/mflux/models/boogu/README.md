# Boogu Image

This directory contains MFLUX's MLX implementation of **Boogu-Image-0.1-Turbo**
([`Boogu/Boogu-Image-0.1-Turbo`](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo)).

Boogu Image Turbo is a 10B text-to-image model from the Boogu team (Apache 2.0),
DMD-distilled down to 4 steps. It leans photographic, with natural lighting and solid
bilingual (English/Chinese) text rendering. Guidance is distilled in: `--guidance` and
`--negative-prompt` are accepted but ignored (the CLI warns when you pass them).

## Example

```sh
mflux-generate-boogu \
  --prompt "A street food vendor at night under paper lanterns, steam rising, neon reflections on wet pavement" \
  --width 768 \
  --height 768 \
  --seed 42 \
  --steps 4 \
  -q 8
```

4 steps is enough up to ~768px; at 1024x1024 use `--steps 8`, where 4 steps
under-resolves fine detail. Quantization (`-q 3|4|5|6|8`) is supported. LoRA flags are
not available for this model.

<details>
<summary>Python API</summary>

```python
from mflux.models.boogu import BooguImage
from mflux.models.common.config import ModelConfig

model = BooguImage(
    model_config=ModelConfig.boogu_image_turbo(),
    quantize=8,
)
image = model.generate_image(
    seed=42,
    prompt="A street food vendor at night under paper lanterns, steam rising, neon reflections on wet pavement",
    width=768,
    height=768,
    num_inference_steps=4,
)
image.save(path="boogu.png")
```

You can also call the steps of `mflux-generate-boogu` from Python. `BooguImageCommand` in `mflux.models.boogu.cli.boogu_image_generate` has `validate(args)`, `load(args)` and `generate(model, args, seed, prompt)`. They work like the Z-Image Turbo steps; the [Z-Image README](../z_image/README.md#z-image-turbo-example) has a full script and the rules for keeping a model loaded.

</details>
