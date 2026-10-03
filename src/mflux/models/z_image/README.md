# Z-Image
This directory contains MFLUX’s MLX implementation of **Z-Image** and **Z-Image-Turbo**.

MFLUX supports [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) from Tongyi Lab (Alibaba). Z-Image is an efficient 6B-parameter image generation model with a single-stream DiT architecture. Z-Image-Turbo delivers high-quality images in just 9 steps, making it one of the fastest open-source models available.

All the standard modes such as img2img, LoRA and quantizations are supported for this model. See the [technical paper](https://arxiv.org/abs/2511.22699) for more details.

ComfyUI LoRAs with fused QKV projections and normalization/bias deltas (`.diff`/`.diff_b`) are supported. Direct deltas require the default baked inference mode; they cannot be used with `--no-bake-lora` or role-controlled training adapters.

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

You can also call the steps of `mflux-generate-z-image-turbo` from Python. `ZImageTurboCommand.load(args)` builds the model; `ZImageTurboCommand.generate(model, args, seed, prompt)` makes one image and returns it unsaved. A script or a UI reuses the command's flag handling (the `--model` check, LoRA options, sizes like `2x`) without copying it. `ZImageTurboCommand.validate(args)` checks a request without loading weights (it runs the `--model` check) and returns the model config; `load()` runs it too. `validate()` reads no files. LoRA names are resolved earlier, when the flags are parsed, so parsing can download a LoRA. The script below takes the command's own flags:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mflux"]
# ///
# Takes the same flags as mflux-generate-z-image-turbo, for example:
#   uv run generate_turbo.py --prompt "A puffin standing on a cliff" --seed 42 43 -q 8
import gc

import mlx.core as mx

from mflux.models.z_image.cli.z_image_turbo_generate import ZImageTurboCommand, build_parser
from mflux.utils.prompt_util import PromptUtil


class PrintProgress:
    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        print(f"seed {seed}: step {t + 1}/{config.num_inference_steps}")


args = build_parser().parse_args()
model = ZImageTurboCommand.load(args)  # once per process
model.callbacks.register(PrintProgress())  # once per loaded model
for seed in args.seed:
    prompt = PromptUtil.read_prompt(args)  # read per seed, as the command does
    image = ZImageTurboCommand.generate(model, args, seed, prompt)
    image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    gc.collect()
    mx.clear_cache()
```

If you keep the model loaded, as a UI or a server does:

- Register your callbacks once per loaded model; there is no unregister call.
- Leave `CallbackManager.register_callbacks` to the command line. The memory saver it adds frees the text encoder during a single-seed run, and the transformer too when that run uses `--low-ram` or `--pid-decode`, so the model cannot be reused after it. Without it the text encoder stays loaded while the image is made, so memory peaks higher than with the command.
- Flags that only `register_callbacks` applies do nothing in this script: `--low-ram`, `--mlx-cache-limit-gb`, `--vae-tiling`, `--vae-tile-size`, `--stepwise-image-output-dir` and `--battery-percentage-stop-limit`.
- A new `--model`, `-q`, `--compute-precision` or LoRA needs a new `load()`. Drop the old model and any of your objects that hold it first (`del model`, then `gc.collect()` and `mx.clear_cache()`), so two sets of weights are never in memory at once.
- Handle one request at a time. The parser reads `sys.argv`, so set it to the request's flags before you call `parse_args()`.
- Save each image before you parse the next request: parsing sets process-wide metadata state, and `--no-exif` stays in effect for the rest of the process.

The other two Z-Image commands have the same steps. `mflux-generate-z-image` is `ZImageCommand` in `mflux.models.z_image.cli.z_image_generate`, and `mflux-generate-z-image-controlnet` is `ZImageTurboControlnetCommand` in `mflux.models.z_image.cli.z_image_turbo_generate_controlnet`. The rules above apply to both, with three differences:

- The base command runs `flow_match_euler_discrete` unless you pass `--scheduler`. Only the command line applies that default, so a script that calls `generate()` sets `args.scheduler` itself.
- The controlnet command's `validate()` also checks the `--control` specs. It raises `ValueError` for a bad spec or for a `--model` without a ControlNet, and `ModelConfigError` (a `ValueError` too) for a `--model` it cannot place.
- The controlnet command keeps the depth, HED and pose detectors loaded for the rest of the process once a control of that type has run. Dropping the model does not free them.
</details>

> [!WARNING]
> Note: Z-Image weights are large (~31GB). Use quantization for smaller sizes.

## Z-Image Turbo ControlNet (Union 2.1)

`mflux-generate-z-image-controlnet` runs Z-Image-Turbo with the [Fun ControlNet Union 2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) from Alibaba PAI. One checkpoint takes five kinds of hint, and every hint is computed locally from an ordinary picture: `canny` and `mlsd` with OpenCV, `depth` with DepthPro, `hed` and `pose` with native MLX detectors (the detector weights download on first use). Give the picture, not a pre-drawn edge map:

```sh
mflux-generate-z-image-controlnet \
  --prompt "a white marble statue of a girl holding a frog, museum lighting" \
  --control "canny:photo.png:0.85" \
  --width 640 \
  --height 368 \
  --steps 20 \
  --seed 3 \
  -q 8
```

- `--control type:path[:strength]`, repeatable, so `--control "pose:a.png:0.8" --control "depth:b.png:0.6"` stacks two hints. The strength is the model card's `control_context_scale`: 0.65 to 1.0 is the range where the picture follows the hint, and 1.0 is the default. `--controlnet-strength` is a global multiplier over every control, 1.0 by default; at 0.4 (the FLUX ControlNet's default, and this command's until 0.19.3) a `0.85` control came out at 0.34 and the composition ignored the hint.
- Width and height are rounded down to multiples of 16, and the control picture is resized to that size, so ask for 640 x 368 rather than 638 x 367 and the hint lines up with the output.
- The 2.1 checkpoint lost part of Turbo's distillation in training, which the model card says outright: at 6 steps the result is soft, at about 20 it is clean. The card also lists 8-step distilled variants of the same ControlNet (`...-Union-2.1-8steps`, `...-2601-8steps`); this command loads the 2.1 file, and the only way to run another one today is a model directory saved with `mflux-save` whose `controlnet/` holds that file and its config.
- `--model` defaults to `z-image-controlnet`, the Turbo transformer plus the Union checkpoint; a local directory saved with `mflux-save` from that model works the same.

<details>
<summary>Python API (ControlNet)</summary>

```python
from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
from mflux.models.z_image.variants.controlnet.z_image_turbo_controlnet import ZImageTurboControlnet

model = ZImageTurboControlnet(quantize=8)
image = model.generate_image(
    seed=3,
    prompt="a white marble statue of a girl holding a frog, museum lighting",
    controls=[ControlSpec(type=ControlType.canny, image_path="photo.png", strength=0.85)],
    controlnet_strength=1.0,  # the API's own default is 0.8
    num_inference_steps=20,
    width=640,
    height=368,
)
image.save("statue.png")
```
</details>

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

