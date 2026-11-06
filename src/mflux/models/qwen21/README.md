# Qwen Image 2.1

For instruction-based single/multiple-reference editing, prefix KV caching, and RGBA output, see [reference editing](reference/README.md) and `uv run mflux-generate-qwen-2.1-edit`. Both commands share the Transformer, VAE encoder/decoder, language decoder, component-loading mechanism, and LoRA mappings. The existing command below retains its text-to-image and strength-based img2img behavior.

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

### Turbo (8 steps)

`Qwen/Qwen-Image-2.1-Turbo` is a distilled checkpoint of the same model. It uses the same architecture and runs in 8 steps without CFG. Select it with `--model qwen-image-2.1-turbo` (aliases: `qwen-2.1-turbo`, `qwen-image-21-turbo`, `qwen-image-turbo-21`). Both `mflux-generate-qwen-2.1` and `mflux-generate-qwen-2.1-edit` accept it:

```sh
mflux-generate-qwen-2.1-edit \
  --model qwen-image-2.1-turbo \
  --prompt "Close-up portrait of a majestic tiger in its natural habitat, detailed fur texture, piercing eyes, natural forest background, soft natural lighting, wildlife photography, photorealistic, high detail, professional wildlife shot" \
  --seed 42 \
  -q 8
```

- The checkpoint stores its 8 sigma values in `model_index.json` (`sample_sigmas`). mflux uses these values without a resolution shift, as the reference pipeline does. The default `linear` scheduler selects this schedule for the Turbo model.
- Use 1024 px or larger. At small sizes such as 512×320, Turbo turns fine detail into a pixel grid. The diffusers reference does the same.
- The defaults are 8 steps and guidance 1. `--steps` accepts any count: other counts sample the 8-node curve at evenly spaced points. Other values for `--guidance`, and `--scheduler viggle_turbo`, stop with an error before the model loads.
- ControlNet: `mflux-generate-qwen-2.1-controlnet --model qwen-image-2.1-turbo-controlnet` runs the Fun ControlNet-Union on the Turbo checkpoint in 8 steps. alibaba-pai trained the branch with the base model. In a canny test at 1280×768, the Turbo output followed the edges as closely as the base model at 20 steps. See [ControlNet](#controlnet).
- LoRAs for Qwen-Image-2.1 load on the Turbo checkpoint. The ausboss Outpaint LoRA v2 kept the original picture in place (kept-area PSNR 37.8 dB, against 20.7 dB without the LoRA). The ausboss Consistency LoRA kept the framing of a watercolor edit.
- Weights: about 32 GB bf16 on disk (14.2 GB transformer, 17.5 GB text encoder in one file, 0.7 GB VAE). A q8 save from the edit command is 17 GB (measured with `du`). That command quantizes the text encoder too.
- Validation: the diffusers reference made a bf16 image at 1024², seed 42. The q8 edit command started from the same initial noise and got PSNR 29.5 dB against that image.

### ControlNet

`mflux-generate-qwen-2.1-controlnet` runs alibaba-pai's [Qwen-Image-2.1-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) next to the base model. One checkpoint reads canny, depth, grayscale, HED, lineart, MLSD, pose and scribble images, so there is no flag for the control type.

```sh
mflux-generate-qwen-2.1-controlnet \
  --prompt "A snowy bakery storefront at dusk, warm light in the windows" \
  --controlnet-image-path canny.png \
  --seed 42 \
  -q 8
```

- The defaults are 40 steps and guidance 1. `--controlnet-strength` scales the control: 1.0 (the default) is the strongest, 0 turns it off.
- Add `--model qwen-image-2.1-turbo-controlnet` to run the same ControlNet on the Turbo checkpoint in 8 steps.
- Without `--width` and `--height`, the output size comes from `--output-resolution` (default 1024) and the aspect ratio of the control image (of the inpaint source when there is no control image). Give a control image at the output size, or pass `--width` and `--height` that match it. A 768×768 canny map gave a sharp image that followed the prompt at 768×768. The same map scaled up to 1024×1024 gave a washed-out image that lost the snow, with both the base and the Turbo checkpoint.
- `--image-path` with `--mask-image` inpaints: white areas of the mask are generated from the prompt, black areas are kept. It works with or without a control image:

```sh
mflux-generate-qwen-2.1-controlnet \
  --model qwen-image-2.1-turbo-controlnet \
  --prompt "A coffee shop entrance with a neon sign reading OPEN" \
  --image-path photo.png \
  --mask-image mask.png \
  --seed 42 \
  -q 8
```

- The output is RGBA, so `--output` must be a PNG, WebP or TIFF file.
- Measured on M5 Max with `-q 8` at 1024×1024: peak memory 27 GB, about 45 s for Turbo and 170 s for the base model, model load included.

### Small text encoder (experimental)

`--small-text-encoder` replaces the Qwen3-VL-8B text encoder with Qwen3.5-0.8B and the 158M text-fusion adapter from [AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit). The adapter maps 6 hidden layers of the 0.8B encoder to the 4096-dim condition of the Qwen-Image-2.1 transformer. The transformer and the VAE stay the stock Qwen-Image-2.1 ones, and mflux does not download the 17.5 GB `text_encoder` folder.

```sh
mflux-generate-qwen-2.1-edit \
  --small-text-encoder \
  --prompt "Change the background to a beach at sunset" \
  --image-paths photo.png \
  --seed 42 \
  -q 8
```

The small text encoder (Qwen3.5-0.8B + the zen-image-edit adapter) on a red-fox prompt across four lighting conditions:

![Small text encoder output](../../assets/qwen_image_21_small_encoder.jpg)

The same red-fox subject with the small and the full Qwen3-VL-8B text encoder:

![Small vs full text encoder](../../assets/qwen_image_21_small_encoder_compare.jpg)

- `mflux-generate-qwen-2.1` and `mflux-generate-qwen-2.1-edit` accept the flag. The edit command also sends the reference images through the 12-layer vision tower of Qwen3.5-0.8B.
- The first run downloads about 2.1 GB: the 1.7 GB encoder and processor files, and the 320 MB `text_fusion.*` byte range of the zen-image-edit transformer. mflux reads that byte range with an HTTP range request and does not download the 14.5 GB transformer. The adapter file goes into the mflux cache (`MFLUX_CACHE_DIR`).
- `-q` does not quantize the small encoder. Its language model runs in bf16. The vision tower and the adapter run in float32, because in bf16 the image features drift away from the reference. The float32 vision tower uses about 200 MB more memory than a bf16 one (402 MB instead of 201 MB).
- `--auto-mask`, `--enhance-prompt` and `--verify` need the Qwen3-VL chat model. With `--small-text-encoder`, they stop with an error.
- A model built with the flag cannot be saved (`save_model()` raises an error). The image metadata records the flag, so `--config-from-metadata` uses it again.
- AiArtLab trained the adapter on Qwen-Image-2.1. Their model card gives a cosine similarity of 0.95 on text positions and 0.97 on vision positions against the native encoder. Prompts must be in English, and numerals in rendered text can come out wrong.
- The zen-image-edit pipeline uses guidance 1.5 with a negative prompt by default. mflux keeps the Qwen-Image-2.1 defaults. To match the zen-image-edit pipeline, add `--guidance 1.5 --negative-prompt "..."`.
- License: the adapter uses the Qwen Research License (the same as Qwen-Image-2.1). Qwen3.5-0.8B uses Apache-2.0.

### img2img

Pass `--image-path` and optionally `--image-strength`, like the other models.

## Training
`mflux-train` trains a LoRA on Qwen-Image-2.1 (the base model, not Turbo) for text to image, with the recipe of ai-toolkit's `qwen_image_2` model. For the folder of images and captions, see the common [training docs](../common/README.md#training-lora).

```sh
mflux-train --config src/mflux/models/common/training/_example/train_qwen_image_21.json
```

[`train_qwen_image_21.json`](../common/training/_example/train_qwen_image_21.json) puts rank 16 (alpha 16) on the attention (`to_q`, `to_k`, `to_v`, `to_out.0`) and the feed-forward (`gate_layer`, `proj`, `out`) of the 32 blocks, the layers ai-toolkit adapts, with its AdamW settings (lr 1e-4, eps 1e-6, weight decay 0.01, gradient clipping at 1.0). The noise level of each step is `sigmoid(N(1, 1))` with no resolution shift. ai-toolkit uses `sigmoid(N(0, 1))`; at 300 steps that gave style LoRAs too weak to show at scale 1, and the mean of 1 fixed them, so `timestep_low`, `timestep_high` and `timestep_type` do not apply. Captions are encoded as prompts are at inference, and 5% of the steps get the empty caption (ai-toolkit's `caption_dropout_rate`). Once the captions are encoded, mflux encodes the preview prompts and lets the text encoder go. Previews run without CFG, as `mflux-generate-qwen-2.1` does by default.

The LoRA is saved with ai-toolkit's keys (`diffusion_model.transformer_blocks.N.attn.to_q.lora_A.weight`, ...), so it loads with `--lora` and in ComfyUI. mflux trains the feed-forward gate and up as two layers; the file stacks them into ComfyUI's single `img_mlp.gate_up` at twice the rank, which gives the same weights.

## Notes

The notes below describe `uv run mflux-generate-qwen-2.1`. The reference-editing command has its own [capabilities and limitations](reference/README.md).

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
- The text prefix K/V is cached per (prompt, resolution) in the text-to-image command too:
  `causal_condition` makes text activations step-independent, so each denoise step runs
  image tokens only against the cached prefix. The cache holds at most the two most recent
  embeddings (the positive and negative CFG prompts); padded prompts recompute the joint
  sequence. Set `use_text_cache = False` on the transformer to force the recompute path.
  The reference-editing command (`uv run mflux-generate-qwen-2.1-edit`) keeps its own prefix cache.
- Q/K norm+rope runs as one stride-aware custom Metal kernel and writes the
  attention-ready `[batch, heads, tokens, head_dim]` layout directly. It supports
  float32/float16/bfloat16 inputs and norm weights, preserving MLX RMSNorm's dtype
  promotion, reduction grouping, and intermediate rounding before rotary embedding.
  Rotary products are rounded separately before addition/subtraction, as in the
  composed path. The fused path
  requires `head_dim` a multiple of 64 up to 512, matching rope tables, and norm
  epsilon `1e-6`; other configurations use the composed implementation.
  If custom Metal kernel construction is unsupported, attention also uses the
  composed path and does not retry construction for the rest of the process.
  `MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE=1` also selects the composed path.
- Step reuse (TeaCache-style, shared across models via `StepCache`): `--step-cache-ratio 0.25`
  (Python: `generate_image(..., step_cache_ratio=0.25)`) skips the transformer on the ~25% of
  denoise steps whose timestep-embedding signal changes least (first/last 10% of the run are
  never skipped) and reuses the previous noise prediction. Measured on M4, 512², 40 steps:
  1.40× at ratio 0.25 (PSNR 27.5 dB, SSIM 0.943 vs uncached), 1.77× at 0.4 (PSNR 25.4 dB,
  SSIM 0.910). Skipping depends only on the sigma schedule, so it is deterministic for a given
  (steps, resolution, ratio), and the ratio is recorded in image metadata. If the selector picks
  two or more steps in a row, all of them reuse the same noise prediction. `--teacache-ratio`
  and `teacache_ratio=` remain as aliases.
- LoRA: `--lora adapter.safetensors 1.0` in either command (PEFT `.default`, bare, `transformer.` and `diffusion_model.` keys, with `lora_A/B`, `lora_down/up` or `lora.down/up` and per-layer `.alpha`; DoRA and mixed full-weight files are not supported). The Comfy-Org 5/8-step and `avg_rank_178` turbo LoRAs and lightx2v Lightning give clean images at their step counts with the default scheduler. The Comfy-Org 6-step v0.2 and Fun-Acc 4-step LoRAs load but expect their own sigma nodes, which the default scheduler doesn't use, and ghost without them. The original alibaba-pai Fun-Acc file keeps 4 per-step `proj_out` heads that mflux can't apply; use the ComfyUI conversion.
- Shared VAE and language implementations live under `qwen21/model`. Adapters preserve each command's existing color/shape, normalization, attention, and precision behavior. The text-only encoder does not instantiate a vision tower; editing adds visual tokens and DeepStack and uses the language decoder output before final RMSNorm.
- Original Hugging Face checkpoints and older text/edit exports remain loadable. The loader translates legacy VAE parameter names without changing tensor values. Text-only exports still omit the vision tower and cannot serve as complete editing checkpoints. Old `qwen21.reference` imports remain available as compatibility exports.
- Not yet supported: PID decoding.
