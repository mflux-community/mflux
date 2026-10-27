# Anima

![Anima Example](../../assets/anima_example.jpg)

This directory contains the MFLUX MLX port of **Anima**
([`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima)), a 2B text-to-image model for anime,
illustration and other non-photographic art from CircleStone Labs and Comfy Org.

Anima is a fine-tune of NVIDIA's Cosmos-Predict2-2B. A Qwen3-0.6B-Base text encoder reads the prompt, and an LLM
adapter inside the transformer turns its hidden states into the 512-token context that the 28 transformer blocks
attend to. The adapter also takes the prompt's T5 token ids. Images are decoded with the Qwen-Image VAE, which
MFLUX already has. The port loads the ComfyUI single files of the model repository, so there is no separate
conversion step.

## Example

```sh
mflux-generate-anima \
  --prompt 'masterpiece, best quality, score_7, safe, 1girl, solo, long hair, silver hair, red eyes, witch hat, black cloak, holding lantern, night, forest, fireflies, looking at viewer, smile' \
  --seed 42
```

Without `--model` the command runs Turbo, the variant the model card recommends to start with: 10 steps at
guidance 1, 1024x1024. The other two variants take more steps and CFG, so they also read the negative prompt:

```sh
mflux-generate-anima \
  --model anima-aesthetic \
  --prompt 'masterpiece, best quality, safe, An anime girl is holding a sign that reads "ANIMA". The background has trees and blue sky with clouds.' \
  --negative-prompt 'worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration' \
  --seed 42
```

| `--model` | File in the repository | Steps | Guidance |
| --- | --- | --- | --- |
| `anima-turbo` (also `anima`, the default) | `anima-turbo-v1.1.safetensors` | 10 | 1.0 |
| `anima-aesthetic` | `anima-aesthetic-v1.1.safetensors` | 30 | 4.0 |
| `anima-base` | `anima-base-v1.0.safetensors` | 30 | 4.0 |

Steps and guidance come from the model card (Turbo: 8 to 12 steps at CFG 1; Aesthetic and Base: 30 to 50 steps
at CFG 4 to 5). Base is the version to train LoRAs on; it has a plainer default style than the other two.

- Width and height must be multiples of 16. The model card names 512x512 to 1536x1536.
- At guidance 1.0 there is no unconditional pass, so a negative prompt has no effect.
- `--scheduler` takes `er_sde` (the default, the model card's choice) or `euler`. The sigma schedule is
  ComfyUI's "simple" schedule on Anima's flow sampling (shift 3).
- `--config-from-metadata` replays the variant, steps, guidance, seed and prompts. It does not replay
  `--scheduler euler`; pass it again.

### Prompting

The model card describes the prompt format in detail. In short:

- Danbooru-style tags (lowercase, spaces instead of underscores), natural language, or both.
- Recommended start of a prompt: `masterpiece, best quality, score_7, safe, `. For Aesthetic, leave the `score_*`
  tags out of both prompts.
- Recommended negative prompt: `worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg
  artifacts, chromatic aberration`.
- Artists take an `@` prefix (`@artist name`). Safety tags are `safe`, `sensitive`, `nsfw` and `explicit`; the card
  advises them in the positive and negative prompts.
- The prompt goes to the encoders as written. ComfyUI's weighting syntax (`(chibi:2)`) is not parsed, and a tag with
  parentheses such as `fern (sousou no frieren)` needs no backslashes.

## Quantization and saving

`-q 8`, `-q 6`, `-q 5` and `-q 4` quantize the transformer blocks. The Qwen3 text encoder, the LLM adapter, the
timestep embedding and the AdaLN modulations stay in bfloat16 at every level: quantized, the text path moves the
context by 40% at 4 bits and garbles lettering, and the modulations set the shift, scale and gate of every token.
`-q 3` does not work with this 2B transformer; the image comes out as noise.

```sh
mflux-save --model anima-aesthetic -q 8 --path ~/models/Anima-aesthetic-mflux-q8
mflux-generate-anima --model ~/models/Anima-aesthetic-mflux-q8 --prompt "masterpiece, best quality, safe, 1girl, smile"
```

A saved folder holds the transformer, the text encoder, the VAE and both tokenizers. The command picks the variant
from the folder name (`aesthetic`, `base`, otherwise Turbo); `--base-model anima-base` names it explicitly.
The three variants have the same weight keys and shapes, so nothing in the weights tells them apart. A renamed
Aesthetic or Base folder therefore runs as Turbo (10 steps, guidance 1) unless you pass `--base-model`.

Saved checkpoints of the three variants are on Hugging Face as `mflux-community/anima-<variant>-mflux-<level>`,
with `<variant>` one of `turbo`, `aesthetic`, `base` and `<level>` one of `q4`, `q5`, `q6`, `q8`, `bf16`. The repo name
picks the variant the same way:

```sh
mflux-generate-anima --model mflux-community/anima-aesthetic-mflux-q4 --prompt "masterpiece, best quality, safe, 1girl, smile"
```

Measured on an M5 Max at 1024x1024 with the defaults above (one warm-up image first):

| Model | Weights | Time per step | Time per image | Peak while denoising | Peak with VAE decode |
| --- | --- | --- | --- | --- | --- |
| Turbo, bf16 | 5.6 GB | 0.42 s | 5.3 s | 6.7 GB | 14.3 GB |
| Turbo, `-q 8` | 4.0 GB | 0.46 s | 5.8 s | 5.1 GB | 12.7 GB |
| Turbo, `-q 4` | 3.1 GB | 0.47 s | 5.8 s | 4.5 GB | 11.8 GB |
| Aesthetic or Base, bf16 | 5.6 GB | 0.96 to 1.05 s | 30 to 33 s | 6.6 GB | 14.3 GB |
| Aesthetic or Base, `-q 8` | 4.0 GB | 1.07 s | 33 s | 5.0 GB | 12.6 GB |

The peak at the end is the Qwen-Image VAE decoding the full image. `--low-ram` decodes it in tiles: Turbo with
`-q 8` then peaks at 4.6 GB.

`--compute-precision float16` runs the attention and feed-forward layers of the blocks in float16. It is meant
for M1 and M2, where bfloat16 is slow; on an M5 Max it is about 20% slower than bf16. The image changes
slightly for the same seed.

Step reuse (`--step-cache-ratio`) is not offered: on Anima's schedule it skips a run of early steps and the
composition breaks.

## LoRA

`--lora` loads Anima LoRAs in the common layouts: kohya's sd-scripts (`lora_unet_blocks_0_self_attn_q_proj`), PEFT
and diffusers names under a `diffusion_model.` or similar prefix, and LyCORIS LoKr. They can target the transformer blocks, the LLM adapter, and the patch and timestep embeddings.

```sh
mflux-generate-anima --model anima-base --lora ~/loras/anima_style.safetensors 0.8 --prompt "..."
```

<details>
<summary>Python API</summary>

```python
from mflux.models.anima import Anima
from mflux.models.common.config.model_config import ModelConfig

model = Anima(model_config=ModelConfig.anima_aesthetic(), quantize=8)
image = model.generate_image(
    seed=42,
    prompt="masterpiece, best quality, safe, 1girl, solo, silver hair, witch hat, night forest, lantern",
    negative_prompt="worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration",
    width=1024,
    height=1024,
)
image.save(path="anima.png")
```

`generate_image` also takes `num_inference_steps`, `guidance` and `scheduler` (`"er_sde"` or `"euler"`). Left
unset, steps and guidance follow the variant. `Anima(...)` takes `model_path`, `lora_paths` and `lora_scales`.

`AnimaCommand` in `mflux.models.anima.cli.anima_generate` has the `validate(args)`, `load(args)` and
`generate(model, args, seed, prompt)` steps of `mflux-generate-anima`.

</details>

## Weights

The model loads from `circlestone-labs/Anima`, and only the files of the variant in use are downloaded: one
transformer file (4.2 GB, the LLM adapter included), `qwen_3_06b_base.safetensors` (1.2 GB) and
`qwen_image_vae.safetensors` (254 MB). The repository has no tokenizers, so the Qwen3 tokenizer comes from
`Qwen/Qwen3-0.6B-Base` (7 MB) and the T5 tokenizer from `google/t5-v1_1-xxl` (0.8 MB). Both give the same token ids
as ComfyUI's bundled tokenizers. The transformer file may use ComfyUI's `model.diffusion_model.` key prefix or the
`net.` prefix of the training format (the Base file has it).

Not included: the older files of the repository (the three preview builds, Turbo v1.0, Aesthetic v1.0 and
v1.0b), image-to-image, and the Anima LLLite control networks.

## Validation

We compared the port with ComfyUI (v0.42.0, `comfy/ldm/anima` and `comfy/ldm/cosmos/predict2.py`) on the real
weights, with ComfyUI's own initial noise and SDE noise fed to MLX:

- In float32 the text encoder output has a cosine similarity of 0.999996 or more with ComfyUI's, the adapter context
  0.99999, one transformer step 0.999998 (0.2% relative error), and the VAE decode a PSNR of 72 dB.
- The Euler and er_sde samplers rebuild ComfyUI's trajectory from its own predictions to a relative error of 5e-8.
- Fed ComfyUI's float32 trajectory, each bfloat16 step differs from it by 2.8% on average (Turbo, worst step 5.5%)
  and 4.5% (Aesthetic with CFG 4, worst 6.8%). ComfyUI's own bfloat16 model differs by 3.9% (worst 17.2%) and 4.5%
  (worst 6.7%). The text encoder and the adapter run in float32 (ComfyUI runs the encoder in float32 on a Mac);
  in bfloat16 the worst step drifted twice as far.
- With the same noise, a kohya LoRA at scales -1 and 1 gives ComfyUI's image; each step differs by 4.1% and 4.4%,
  as without the LoRA.
- Weight-free tests compare the transformer and the LLM adapter with a copy of the reference modules at toy sizes, and
  the samplers with ComfyUI's on a toy denoiser.
