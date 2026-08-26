# Z-Image
This directory contains MFLUX’s MLX implementation of **Z-Image** and **Z-Image-Turbo**.

MFLUX supports [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) from Tongyi Lab (Alibaba). Z-Image is an efficient 6B-parameter image generation model with a single-stream DiT architecture. Z-Image-Turbo delivers high-quality images in just 9 steps, making it one of the fastest open-source models available.

All the standard modes such as img2img, LoRA and quantizations are supported for this model. See the [technical paper](https://arxiv.org/abs/2511.22699) for more details.

![Z-Image-Turbo Example](../../assets/z_image_turbo_example.jpg)

## Z-Image (Base) Example
The following generates with the base Z-Image model. Base (non-distilled) Z-Image uses more steps than the turbo model:

> [!WARNING]
> Base (non-distilled) Z-Image is typically slower and worse for general image editing, but can be successfully used for training.

```sh
mflux-generate-z-image \
  --prompt "A red fox resting in fresh snow under soft winter light, detailed fur, gentle bokeh, natural color grading." \
  --width 720 \
  --height 1280 \
  --seed 42 \
  --steps 50 \
  --guidance 4
```

<details>
<summary>Python API (Base)</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImage

model = ZImage(
    model_config=ModelConfig.z_image(),
    model_path="Tongyi-MAI/Z-Image",
)
image = model.generate_image(
    seed=42,
    prompt="Two smiling friends posing for a casual indoor portrait, soft natural light, shallow depth of field.",
    num_inference_steps=50,
    width=720,
    height=1280,
    guidance=4.0,
    negative_prompt="",
)
image.save("z_image_base.png")
```
</details>

## Z-Image Turbo Example
The following uses the pre-quantized 4-bit model from [filipstrand/Z-Image-Turbo-mflux-4bit](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit) to generate a vibrant 1960s style image with a LoRA adapter [Technically Color](https://huggingface.co/renderartist/Technically-Color-Z-Image-Turbo) for enhanced film color:

```sh
mflux-generate-z-image-turbo \
  --model filipstrand/Z-Image-Turbo-mflux-4bit \
  --prompt "t3chnic4lly vibrant 1960s close-up of a woman sitting under a tree in a blue skirt and white blouse, she has blonde wavy short hair and a smile with green eyes lake scene by a garden with flowers in the foreground 1960s style film She's holding her hand out there is a small smooth frog in her palm, she's making eye contact with the toad." \
  --width 1280 \
  --height 720 \
  --seed 456 \
  --steps 9 \
  --lora-paths renderartist/Technically-Color-Z-Image-Turbo \
  --lora-scales 0.5
```

<details>
<summary>Python API (Turbo)</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImage

model = ZImage(
    model_config=ModelConfig.z_image_turbo(),
    model_path="filipstrand/Z-Image-Turbo-mflux-4bit",
    lora_paths=["renderartist/Technically-Color-Z-Image-Turbo"],
    lora_scales=[0.5],
)
image = model.generate_image(
    seed=456,
    prompt="t3chnic4lly vibrant 1960s close-up of a woman sitting under a tree in a blue skirt and white blouse, she has blonde wavy short hair and a smile with green eyes lake scene by a garden with flowers in the foreground 1960s style film She's holding her hand out there is a small smooth frog in her palm, she's making eye contact with the toad.",
    num_inference_steps=9,
    width=1280,
    height=720,
)
image.save("z_image_turbo.png")
```
</details>

> [!WARNING]
> Note: Z-Image weights are large (~31GB). Use quantization for smaller sizes.

## Training

Use `mflux-train` with a training config that targets `z-image` or `z-image-turbo`. We automatically load the Z-Image Turbo training adapter ([ostris/zimage_turbo_training_adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter)) only when training the turbo model; base Z-Image training does not use the assistant LoRA. You can start from the [example config](../common/training/_example/train.json). For the data/images folder layout, see the common training docs ([Training (LoRA)](../common/README.md#training-lora)).

Example:

```json
{
  "model": "z-image-turbo",
  "data": "images/",
  "seed": 4,
  "steps": 9,
  "guidance": 0.0,
  "quantize": 8,
  "training_loop": { "num_epochs": 1, "batch_size": 1, "timestep_low": 4, "timestep_high": 9 },
  "optimizer": { "name": "AdamW", "learning_rate": 1e-4 },
  "checkpoint": { "output_path": "training", "save_frequency": 30 },
  "monitoring": {
    "plot_frequency": 20,
    "generate_image_frequency": 30
  },
  "lora_layers": {
    "targets": [
      { "module_path": "layers.{block}.attention.to_q", "blocks": { "start": 15, "end": 30 }, "rank": 8 },
      { "module_path": "layers.{block}.attention.to_k", "blocks": { "start": 15, "end": 30 }, "rank": 8 },
      { "module_path": "layers.{block}.attention.to_v", "blocks": { "start": 15, "end": 30 }, "rank": 8 }
    ]
  }
}
```

Run training:

```sh
mflux-train --config /path/to/train_z_image.json
```

*For a Swift MLX implementation of Z-Image, see [zimage.swift](https://github.com/mzbac/zimage.swift) by [@mzbac](https://github.com/mzbac).*

