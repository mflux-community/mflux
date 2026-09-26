# Qwen-Image-2.1 edit validation

How the edit path in this PR was verified, on real weights and weight-free. Numbers below
were measured on this contributor's machine (Apple M2 Ultra, 128 GB unified memory); they
describe the tested checkpoint and inputs, not a quality benchmark.

- Checkpoint: `Qwen/Qwen-Image-2.1` (local snapshot, quantization q8 unless noted).
- Environment: MLX nightly of 2026-09, Python 3.13, macOS 26.
- The reference implementation used for semantic comparison is diffusers'
  `QwenImage21Pipeline` (`pipelines/qwenimage21/pipeline_qwenimage21.py`) and transformers'
  `Qwen3VLForConditionalGeneration`, read side-by-side with the port during development.

## Stage-parity methodology (transformers, real weights)

The text encoder was ported against the real `Qwen3VLForConditionalGeneration` in stages,
isolating one boundary at a time with throwaway comparison scripts (not shipped; each needs
the 33 GB checkpoint):

1. **Preprocessing**: our numpy `Qwen21ImageProcessor` (patchify/merge/normalize) vs
   transformers' `AutoImageProcessor` output — pixel values and `image_grid_thw` agree.
2. **Vision tower**: our `Qwen3VLVisionModel` forward vs `model.visual` — image embeds and
   the three deepstack features compared elementwise.
3. **Deepstack injection**: block-8 hidden states with/without deepstack injection vs
   transformers, isolating the injection positions and the gather-index expansion.
4. **Full encoder**: `forward_vl` hidden states vs transformers' pre-final-norm output
   (note: `transformers>=5` returns post-norm hidden states from the standard forward;
   the comparison used the layer-before-norm activation, which is what the diffusion
   transformer was trained on).

These comparisons are how three port bugs were found and fixed before any image was ever
generated: an MLP weight-mapping pattern that never matched (`{prefix}.{param}` vs
`{prefix}.mlp.{param}` — MLP weights stayed at random init), an inverted anti-causal
attention comparison, and text tokens sharing one block id (`-1 == -1`) in the block-causal
mask. The layout math (mrope positions, segmented prefill, zero-timestep modulation) is
covered weight-free in `tests/image_generation/test_qwen_image_21_edit_transformer.py`,
including exact equality between the edit path and the t2i path when no condition images
are given.

**Honest gap**: unlike #741, we did not ship a component-level fp32 numerical-diffusers
parity harness with pinned revisions; the comparisons above were interactive and are
described qualitatively. The stage isolation list is the reproduction recipe.

## Performance measurements

| Measurement | Result |
|---|---|
| t2i q8, 1024², 40 steps | 7.22 s/step, 289.5 s/image |
| Edit, 20 steps, 1024², q8 | ~6.7–7.6 s/step (varies with reference count) |
| Model load (page-cache warm, mmap) | ~1.2 s; first cold load dominated by disk |
| Prefix KV cache, 3-ref 1024² edit | 28 → 7.7 s/step (**3.6x**), pixel-identical vs uncached |
| Prefix KV cache (fxd0h, M5 Max, 512² 1-ref 8-step) | 14.9 → 8.9 s (**1.67x**), diff 0.06/255 |
| Step cache (threshold 0.12), 1024² 20 steps | **1.21x** end-to-end (~1.5x denoising-only; run included a grounding prefill) |
| 20 vs 40 steps on edits | visually indistinguishable on tested prompts; api_server default set to 20 |

Step-cache output is close to but not pixel-identical with the uncached trajectory (block
reuse is an approximation by design); it is opt-in.

## Capability validation (real weights, q8, 20 steps, 1024², seed 42)

Validation image: a 1024² portrait (qipao dress), `outputs/api/0525ceae0f82.png` locally.

| Capability | Measurement |
|---|---|
| Inpaint (manual mask) | mean pixel diff outside mask vs original **0.0** (bitwise); face/pose/background unchanged, dress repainted |
| Auto-mask grounding | "the dress" → bbox `[359, 168, 580, 783]` (0–1000 normalized convention), lands on the dress; box grown 3%/side + feathered |
| Auto-mask + step cache run | unmasked diff 0.007/255 (wider feathered box); 110 s vs 134 s baseline |
| Edit strength | mean diff vs original: **0.3 → 3.77** (subtle retouch, composition identical), 0.6 → 3.08, 1.0 → 22.46 |
| Prompt rewrite | 4.7 s; terse instruction → 40-word description with explicit preservation constraints; the rewritten satin/embroidery detail is visible in the A/B output |
| Self-verification | correct structured verdicts on pass cases; an engineered "impossible" case turned out feasible (model replaced the background) and the verifier correctly reported it applied |
| RGBA output | 20-step sticker: 86.6% of pixels alpha < 32 (transparent background), 13% > 223 (opaque subject) |
| LM head sanity | "capital of France" → "Paris" (this is how the untied `lm_head.weight` omission was caught — the first grounding attempt returned garbage) |

## Boundary testing (weight-free, real inputs)

`test_qwen_image_21_edit_exif.py`, `test_qwen_image_21_edit_dimensions.py`, and the
real-input matrix run before them:

- EXIF orientation 6 reaches both encoders rotated to display orientation (fixture reloads
  from saved bytes — in-memory `exif_transpose` is a no-op).
- All PIL modes (L/LA/RGBA/P+alpha/CMYK/I;16/1-bit) flow through both encoders; multi-image
  mixed-mode sets build consistent layouts.
- Extreme aspect ratios: 300:1 and 8192:1 inputs raise a clear up-front `ValueError`
  instead of crashing inside vision preprocessing or rounding a side to zero.
- Partial dimensions: an explicit axis is honored per-axis; CLI auto/scale/integer flag
  combinations (including explicit `1x` = source size) are pinned in tests.
- CLI invalid arguments (bad strength/scheduler/guidance, missing images, RGBA-to-JPEG)
  fail in `validate_args` **before** the 33 GB model load.

## Not verified / known limitations

- No pinned-revision diffusers numerical parity harness (see honest gap above).
- Grounding boxes are box-level, not segmentation-level; edges are feathered and the box
  is grown 3% per side, which can touch nearby content.
- Self-verification is a coarse instruction-applied / rest-preserved check, not an
  aesthetic judge.
- Step cache trades exactness for speed by design.
- The `strength` interpolation is the standard flow-matching img2img start; no claim of
  equivalence to any official edit-strength schedule (the reference pipeline has none).
