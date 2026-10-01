# SeedVR2
This directory contains MFLUX’s MLX implementation of the **SeedVR2** upscaler.

SeedVR2 (3B) is a dedicated diffusion-based super-resolution model based on https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler. It is designed to be fast (often 1-step) and highly faithful to the original image. Unlike the ControlNet-based upscaler, it does not require a text prompt.

SeedVR2 is more recent and the preferred method for high-fidelity upscaling and is much faster than the controlnet-based upscaler.

![SeedVR2 Upscale Comparison](../../assets/upscale_seedvr2_comparison.png)

## Upscale

```sh
mflux-upscale-seedvr2 \
  --image-path "input.png" \
  --resolution 2160 \
  --softness 0.5
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.seedvr2 import SeedVR2

model = SeedVR2(model_config=ModelConfig.seedvr2_3b())
image = model.generate_image(
    seed=42,
    image_path="input.png",
    resolution=2160,
    softness=0.5,
)
image.save("input_upscaled.png")
```
</details>

This will upscale the image such that the shortest side is 2160 pixels while maintaining the aspect ratio. If `--model` is omitted, MFLUX defaults to `seedvr2-3b`. Pass `--model seedvr2-7b` to use the 7B model. 

Instead of specifying a target resolution, you can also use `--resolution 2x` or `--resolution 3x` to upscale by a factor of 2 or 3 respectively.

You can also adjust the `--softness` parameter (0.0 to 1.0) to control input pre-downsampling, which can help achieve smoother upscaling results. A value of 0.0 (default) disables pre-downsampling, while higher values up to 1.0 increase the downsampling factor (up to 8x internally) before upscaling. A value of `0.5` is often a good starting point.

> [!NOTE]
> Upscaling to very large resolutions can require a lot of memory. If you run into memory pressure, try `--low-ram` or set an MLX cache limit with `--mlx-cache-limit-gb 16` (replace `16` with a value that fits your machine).

## Save a Quantized Copy

Use `mflux-save` to write a quantized copy of the 3B or the 7B model to disk. Then give that directory to `--model`.

```sh
mflux-save --model seedvr2-7b --quantize 8 --path ~/models/seedvr2-7b-q8
mflux-upscale-seedvr2 --model ~/models/seedvr2-7b-q8 --image-path input.png --resolution 2160
```

The upscaler reads the variant (3B or 7B) from the saved weights. The name of the directory has no effect.

## Upscale a Directory

Pass a directory to `--image-path` to upscale every image inside.

> [!WARNING]
> `--low-ram` is not supported when upscaling a directory. That mode offloads the transformer after each image, which conflicts with the folder-processing flow.

```sh
mflux-upscale-seedvr2 \
  --image-path "./inputs" \
  --resolution 2160 \
  --softness 0.5
```

<details>
<summary>Python API</summary>

```python
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.seedvr2 import SeedVR2

model = SeedVR2(model_config=ModelConfig.seedvr2_3b())
for image_path in sorted(Path("./inputs").iterdir()):
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        continue
    image = model.generate_image(
        seed=42,
        image_path=image_path,
        resolution=2160,
        softness=0.5,
    )
    image.save(image_path.with_stem(f"{image_path.stem}_upscaled"))
```
</details>

<details>
<summary>🛠️ <strong>Example: Generating and Upscaling with Z-Image Turbo</strong></summary>

The comparison image above was produced by first generating a base image using **Z-Image Turbo** and then upscaling it using **SeedVR2**.

**1. Generate the base image**

```sh
mflux-generate-z-image-turbo \
  --prompt "class1cpa1nt a prestigious candlelit banquet table in a high-ceilinged palace hall. The scene features a bottle of \"Z-Image Vintage Select\" beside a sparkling crystal decanter. The table is overflowing with luxury: golden plates, silk napkins, and a centerpiece of dark red roses. Fine details of the wood grain on the table and the reflection of a chandelier in the polished surfaces. The lighting is dramatic and warm, reminiscent of Rembrandt. Masterful oil painting with aged texture and crackle glaze." \
  -q 8 \
  --steps 9 \
  --width 768 \
  --height 336 \
  --seed 42 \
  --lora-paths renderartist/Classic-Painting-Z-Image-Turbo-LoRA \
  --lora-scales 0.5 \
  --output image.png
```

<details>
<summary>Python API</summary>

```python
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(
    quantize=8,
    lora_paths=["renderartist/Classic-Painting-Z-Image-Turbo-LoRA"],
    lora_scales=[0.5],
)
image = model.generate_image(
    seed=42,
    prompt='class1cpa1nt a prestigious candlelit banquet table in a high-ceilinged palace hall. The scene features a bottle of "Z-Image Vintage Select" beside a sparkling crystal decanter. The table is overflowing with luxury: golden plates, silk napkins, and a centerpiece of dark red roses. Fine details of the wood grain on the table and the reflection of a chandelier in the polished surfaces. The lighting is dramatic and warm, reminiscent of Rembrandt. Masterful oil painting with aged texture and crackle glaze.',
    num_inference_steps=9,
    width=768,
    height=336,
)
image.save("image.png")
```
</details>

**2. Upscale 3x using SeedVR2**

```sh
mflux-upscale-seedvr2 \
  --image-path image.png \
  --resolution 3x \
  --softness 0.5
```

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.seedvr2 import SeedVR2
from mflux.utils.scale_factor import ScaleFactor

model = SeedVR2(model_config=ModelConfig.seedvr2_3b())
image = model.generate_image(
    seed=42,
    image_path="image.png",
    resolution=ScaleFactor.parse("3x"),
    softness=0.5,
)
image.save("image_upscaled.png")
```
</details>

</details>
