# Qwen Image 2.1

MFLUX’s MLX implementation of **Qwen-Image-2.1** (`Qwen/Qwen-Image-2.1`), the second-generation
Qwen Image text-to-image model.

![Qwen Image 2.1 showcase](../../assets/qwen_image_21_example.jpg)

Qwen Image 2.1 is a single-stream, block-causal transformer (7.1B) paired with a new 64-channel
causal VAE (16× spatial compression, one latent token per 16×16 pixel tile) and a Qwen3-VL text
encoder. The joint text/image sequence is causal overall while the target image block attends
bidirectionally to itself, and text tokens are conditioned at `t = 0` (`causal_condition`), which
is what makes their activations independent of the denoising step.

The recommended sampling defaults are **40 steps and no guidance** (guidance 1.0), which is what
`mflux-generate-qwen-2.1` uses out of the box.

### Example

```sh
mflux-generate-qwen-2.1 \
  --prompt "Close-up portrait of a majestic tiger in its natural habitat, detailed fur texture, piercing eyes, natural forest background, soft natural lighting, wildlife photography, photorealistic, high detail, professional wildlife shot" \
  --steps 40 \
  --seed 42
```

On Apple Silicon the bf16 default is both the fastest and the most accurate path
(measured on M5 Max: ~1.5 s/step at 1024², ~78 s for a full 40-step generation —
faster than the diffusers MPS reference at ~85 s, with channel correlation
0.989–0.992 against it). Peak memory is ~46 GB (the Qwen3-VL text encoder,
~17.5 GB bf16, is never quantized and stays resident), so a 64 GB machine is
the comfortable default; use `-q 8` when memory is tighter.

<details>
<summary>Example with quantization</summary>

```sh
mflux-generate-qwen-2.1 \
  --prompt "Close-up portrait of a majestic tiger in its natural habitat, detailed fur texture, piercing eyes, natural forest background, soft natural lighting, wildlife photography, photorealistic, high detail, professional wildlife shot" \
  --steps 40 \
  --seed 42 \
  -q 8
```
</details>
<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

model = QwenImage21(
    quantize=None,
    model_config=ModelConfig.qwen_image_21(),
)
image = model.generate_image(
    seed=42,
    prompt="Close-up portrait of a majestic tiger in its natural habitat, detailed fur texture, piercing eyes, natural forest background, soft natural lighting, wildlife photography, photorealistic, high detail, professional wildlife shot",
    num_inference_steps=40,
    width=1024,
    height=1024,
)
image.save("qwen21_tiger.png")
```
</details>

### True CFG (optional)

2.1 is trained to be sampled without guidance. If you want classifier-free guidance anyway,
pass a negative prompt together with `--guidance > 1`:

```sh
mflux-generate-qwen-2.1 --prompt "..." --negative-prompt "blurry, low quality" --guidance 2.5
```

With no negative prompt (or `--guidance 1.0`, the default) the negative pass is skipped entirely.

### img2img

Pass `--image-path` and optionally `--image-strength`, like the other models.

## Notes

- Weights: `Qwen/Qwen-Image-2.1` (~33 GB bf16 on disk: 14.2 GB transformer, 17.5 GB text encoder,
  1.4 GB VAE). The text encoder is kept in bf16 like the 1.x port; quantization applies to the
  transformer and VAE.
- The VAE has 4 input/output channels (RGBA). The alpha channel carries edit masks, not image
  content — decode returns RGB only. Encoding pads a solid alpha channel for img2img.
- The upstream VAE registers per-frame `time_conv` layers that only run in the chunked video
  path; mflux runs the single-image path where they are dead weights, so they are intentionally
  not mapped or loaded.
- The prompt template is a raw string (not `apply_chat_template`) with the system-role tokens
  dropped from the final hidden states, matching the reference pipeline exactly.
- The text prefix KV cache (valid because `causal_condition` makes text activations
  step-independent) is a planned optimization; the current port recomputes the prefix each step.
- LoRA: `--lora adapter.safetensors 1.0` (PEFT `.default` format).
- Not yet supported: the edit/instruction variant (needs the Qwen3-VL vision tower)
  and PID decoding.
