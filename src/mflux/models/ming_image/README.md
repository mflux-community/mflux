# Ming-Image

This directory contains MFLUX's MLX implementation of **inclusionAI Ming-Image-0.1-Design**, a
text-to-image model tuned for graphic design (posters, cards, UI, typography) that outputs
**RGBA** images.

The checkpoint is three models glued together:

| Part | Architecture | Size |
|---|---|---|
| Text encoder | Ling-mini-2.0 (`bailing_moe_v2`): 20 layers, 256 experts (top-8, 1 shared), 3D `video_rope` | ~16B (1.4B active) |
| Connector | Qwen2-1.5B stack run bidirectionally over 256 learned query tokens | 1.5B |
| DiT | Z-Image single-stream DiT (30 layers + 2+2 refiners, dim 3840) | 6.15B |
| VAE | Qwen-Image VAE retrained for RGBA | 0.25B |

The prompt is encoded twice: the 256 query tokens' final states go through the connector
(`cap_feats`), and the prompt tokens' hidden states from layers 5, 12 and 20 are projected
straight to DiT width (`cap_feats_2`). The DiT reuses the Z-Image blocks and weight layout.
The Qwen2.5 ViT inside the checkpoint's `mllm/` folder is not used for text-to-image and is
never loaded.

## Example

```sh
mflux-generate-ming \
  --model path/to/ming-image-q8-te5 \
  --prompt "A minimalist poster that says HELLO MAC" \
  --width 1024 --height 1024 \
  --steps 12 \
  --seed 42
```

Output PNGs keep the alpha channel; pass `--flatten-alpha` for an RGB image on white.

Defaults follow the official pipeline: 12 steps (the schedule's last step has sigma 0, so 11
DiT evaluations), guidance 1.0 (CFG off; `--guidance` > 1 uses the official zeroed-condition
negative), and the flow-match Euler schedule with the checkpoint's static shift of 6. The
model was trained on aspect-ratio buckets around 1024² and 2048²; any size that is a multiple
of 16 works.

## Quantization

The text encoder is quantized separately from the DiT:

```sh
# 8-bit DiT and connector, 4-bit MoE text encoder (~17 GB on disk)
mflux-generate-ming --prompt "..." -q 8 --text-encoder-quantize 4
```

Keep the DiT at 8 bits: at 4 bits its typography visibly degrades (broken strokes in large
headline letters), which matters for a design model. The text encoder tolerates 4 bits well.
The routers, norms, learned query tokens and projection heads stay in full precision.

Quantizing from the original checkpoint needs the bf16 weights in memory (~50 GB with the
unused ViT). On a smaller machine, convert once elsewhere and load the saved model:

```sh
mflux-save --model ming-image-design --path ming-q8 -q 8
```

(`mflux-save` applies one level to every component; `MingImage(quantize=8,
text_encoder_quantize=4).save_model(path)` writes the mixed variant, and a saved mixed
checkpoint reloads each layer at the precision it was stored with.)

The CLI encodes the prompt once and frees the text encoder before denoising, so peak memory
is the larger of the two stages rather than their sum.

| Text encoder bits (DiT q8) | Size on disk | Fidelity vs official (same noise) |
|---|---|---|
| 8 | 25.2 GB | matches |
| 6 | 21.2 GB | matches |
| 5 | 19.3 GB | matches (recommended for 24 GB Macs) |
| 4 | 17.3 GB | clean typography, but scene details drift |

## Performance

Base M4 Mac mini (10-core GPU, 24 GB), q8 DiT + q5 text encoder, 12 steps:

| Stage | Time | Peak GPU memory |
|---|---|---|
| Prompt encode (then released) | 4 s warm, 18 s cold disk cache | 12.5 GB |
| 512² denoise + decode | 63 s | 9.8 GB |
| 1024² denoise + decode | 4–4.5 min | 14.5 GB |

The DiT is compute-bound on this GPU (~20 s per step at 1024²); `mx.compile` does not change it.

## Fidelity notes

Verified stage by stage against tensors dumped from the official PyTorch pipeline (cosine
similarity, bf16): DiT step outputs 0.9999+, VAE 0.99999, connector 0.9999; the text
encoder's hidden states match to within the official model's own eager-vs-flash-attention
spread. Two upstream behaviours are reproduced on purpose:

- The official pipeline encodes the prompt under `torch.autocast(bfloat16)`, which makes the
  MoE router's logits, sigmoid scores and `scores + expert_bias` bf16 (group sums and mixing
  weights stay float32). With expert biases near 17.25 most experts tie at bf16 resolution and
  `torch.topk` resolves ties towards the lower index. `LingGate` reproduces this exactly;
  routing in float32 selects noticeably different experts.
- The scheduler's `use_dynamic_shifting` override in the official code never reaches the
  frozen config, so the effective schedule is the static shift of 6.

<details>
<summary>Python API</summary>

```python
from mflux.models.ming_image import MingImage

model = MingImage(model_path="path/to/ming-image-q8-te4")
image = model.generate_image(
    seed=42,
    prompt="A minimalist poster that says HELLO MAC",
    width=1024,
    height=1024,
    num_inference_steps=12,
)
image.save(path="ming.png")
```

</details>
