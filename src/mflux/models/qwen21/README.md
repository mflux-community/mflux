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

### Editing

Qwen Image 2.1 is a unified generation + editing model: editing is the same pipeline with one
or more condition images. The Qwen3-VL text encoder reads each condition image as vision
context (its vision tower ships inside the `text_encoder` shards, so no extra download is
needed) while the VAE encodes it into latent tokens prepended to the noise sequence.

```sh
mflux-generate-qwen-2.1-edit \
  --image-paths input.png \
  --prompt "Change the background to a sunset beach" \
  --steps 40 \
  --seed 42
```

Multiple condition images are referenced as `Image 1`, `Image 2`, ... in the prompt:

```sh
mflux-generate-qwen-2.1-edit \
  --image-paths photo1.png photo2.png \
  --prompt "Put the person from Image 1 into the scene from Image 2" \
  --steps 40
```

RGBA condition images keep their alpha for the VAE (edit masks); the vision encoder sees a
white-composited copy. Output dimensions default to the last condition image's aspect ratio
at ~1MP; passing `--width`/`--height` explicitly keeps the given axis and derives only the
missing one (floored to /16 multiples like everywhere in mflux). Condition aspect ratios
beyond 200:1 are rejected up front — the vision encoder cannot process them. At most 10
condition images are supported (matching the reference pipeline). Invalid arguments fail
before the model loads. `--rgba-output` keeps the decoder's alpha channel for transparent
output (PNG/WebP/TIFF only).

### Prefix KV cache

Editing steps after the first only recompute the target-image queries: because
`causal_condition` modulates text and reference tokens from `t = 0`, their activations are
step-independent, so the first step prefills a per-layer K/V cache of the text + reference
prefix and later steps attend it. `QwenImage21Edit.generate_image(..., use_kv_cache=True)`
(enabled by default) measured 3.6x faster on a three-reference 1024² edit (28 -> 7.7 s/step
on M2 Ultra bf16) with pixel-identical output against the uncached path.

### Inpainting and outpainting

Pass a mask aligned with the first condition image (`--mask-image mask.png`, or the
`mask_image` Python/API parameter): white marks the region to repaint, black the region to
preserve. During denoising the unmasked latent tokens are pulled onto the reference's own
noised trajectory each step, and the decoded result is composited with the original pixels
outside the mask, so untouched content stays exactly the original. Outpainting is the same
mechanism: paste the image onto a larger canvas, mask the added border, and prompt for the
surroundings.

### Auto-masking (visual grounding)

Instead of drawing the mask, describe the object: `--auto-mask "the dress"`. The edit model
already holds a full Qwen3-VL language model (the encoder ships its generation head tied to
the embeddings), so the variant asks it where the named object is, parses the returned
bounding box, feathers it into a mask, and feeds the inpainting path above. The reply is
parsed as absolute coordinates of the (resized) image it was shown; an unparseable reply
raises an error pointing at `mask_image` instead.

### Step cache

`--use-step-cache` (Python: `use_step_cache=True`) enables first-block step skipping on top
of the prefix KV cache: block 0 always runs, and when a relative-L1 signal of consecutive
steps stays under `step_cache_threshold` (default 0.12) the remaining blocks reuse the
previous step's hidden state, with the output norm re-applied to the current timestep.
Nearby denoising steps differ little, so most steps skip 31 of 32 blocks; output stays
close to the baseline but is not pixel-identical. Requires `use_kv_cache=True`.

### Edit strength

`--strength` (Python: `strength=`) in (0, 1] controls how far the edit re-denoises.
`strength=1.0` (default) starts from pure noise — the fullest reimagining. Lower values
start partway down the sigma schedule from the reference's own latent noised to that
sigma, exactly like mflux's img2img: `strength=0.3` runs the last 30% of steps for subtle
retouching, `0.7` for a strong but composition-preserving change. Composes with inpainting
masks (the mask still pins unmasked pixels to the original).

### Prompt enhancement and self-verification

The edit model holds a complete Qwen3-VL in memory — including the untied `lm_head` the
checkpoint ships — so two official-recipe capabilities run with zero extra weights:

- `--enhance-prompt` rewrites a terse instruction into a detailed descriptive prompt
  before encoding, following the official `prompt_rewrite` serving recipe
  (QwenLM/Qwen-Image-2.1). The reply is parsed for `{"rewritten_prompt": ...}`; on any
  failure the original instruction is used unchanged.
- `--verify` shows the original and the result to the same Qwen3-VL after generating and
  records a structured verdict (`instruction_applied`, `outside_unchanged`) on the
  returned image's `.verification`. `--verify-retries N` regenerates with a new seed at
  most N times when the verdict fails.

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
- The prefix KV cache (valid because `causal_condition` makes text and reference activations
  step-independent) is implemented for the edit path: the first step prefills a per-layer
  K/V cache of the text + reference prefix and later steps recompute only the target queries.
  The t2i path still recomputes the prefix each step.
- Not yet supported: LoRA mappings, and PID decoding.
