# PRD: Z-Image ANE Split, Stage 3 (Full DiT, Packaging, CLI)

## Goal
Turn the Stage 2 single-layer proof into an **opt-in mflux feature**: every large DiT linear layer is split between the GPU (MLX) and the Apple Neural Engine (ANE, int8 via Core ML). The feature has to be packaged, cached and controllable from the CLI.

**Target:** ≥1.3× per-step speed-up against q8 at 1024² on an M4 Max, with q8-class image quality as judged by Scriv by eye, and memory no higher than q8 + 10%.

## Background (Stages 1–2, branch `experiment/zit-on-ane` on the fork)
- **Linear layers are 79–80% of DiT block time;** attention is ~16%, everything else ~5% (Agent 2's investigation). MLX bf16 matmul is near the GPU's practical ceiling, so the ANE is the remaining lever.
- **ANE as a 1×1 `Conv2d`** in channels-first layout `(1, C, 1, S)`, with weights baked in (V2), reaches ~14 TFLOPS in W8A16. For w2 (inner dimension 10,240) it needs a 4×2560 inner-dimension split with each chunk's weights baked separately.
- **Concurrency:** the GPU and ANE overlap with no GPU slowdown. In the synthetic test, 50:50 gave 1.97× on w1.
- **Stage 2, in-process:** the w1 layer split runs 19.8 ms against 31.5 ms for q8 on the GPU alone (**~1.59×**), cosine 0.999994. Projected per-step gain: **~1.37×** before fused calls.
- **W8A8** (int8 activations too): 1.47× bf16 on the ANE alone, slightly lower accuracy (max diff ~4% against ~1%). **Kept as a first-class option**, judged by eye.
- **fp16 overflow:** w2 outputs reach ~80,000, above fp16's maximum of 65,504. Dividing inputs by 32 before the ANE and multiplying outputs by 32 after (the repo's existing `FeedForward.FLOAT16_HEADROOM`) fixes it.

## Decisions already made
- **Source models:** bf16 or q8 checkpoints only. q4 or other formats with `--ane` are refused with a clear message.
- **One model in memory:** at load, each layer is split. The GPU keeps its columns (bf16 or q8); the ANE columns are dropped from MLX and served from Core ML.
- **Default precision W8A16; W8A8 opt-in.**
- **Row padding, not image resizing:** ANE inputs are padded up to a token-count tier and the extra output rows discarded. A linear layer treats each row independently, so the output is unchanged.
- **Distributed artifacts are `.mlpackage`; the compiled `.mlmodelc` is a local cache only.**
- **coremltools** is a dependency with the marker `sys_platform == "darwin"`.

---

## Stage 3a: Infrastructure tests (go/no-go before integration)
Run each with synthetic weights at real shapes. Report speed, ANE placement (`aneplan`, plus `powermetrics` run by Scriv) and accuracy.

1. **Fused shapes:** QKV fused (3840 → 11,520) and w1+w3 fused (3840 → 20,480), compared with separate calls, both alone and in a concurrency split. Goal: 4 ANE calls per block instead of 7.
2. **Concurrency splits for every shape:** 3840² (o), QKV fused, w1w3 fused and w2 (K-split), at GPU:ANE 60:40, 50:50, 45:55 and 40:60. Run with the **GPU half in q8 and in bf16**. Pick the best ratio per shape.
3. **Enumerated shapes:** one function accepting token counts {4608, 6528, 8320, 16128}. Does it stay on the ANE, and at what speed against fixed-shape models? If yes, there are **no per-tier files**.
4. **Multifunction packages:** many functions in one `.mlpackage` (per block, or per tier). Check that weights are deduplicated, how fast each function loads, and the runtime speed of each function's `MLModel` (via `functionName`) against standalone packages. Include W8A16 and W8A8 functions sharing weights.
5. **Group-wise int8 matching MLX q8:** an ANE weight format that reproduces MLX's 8-bit group-wise scale/bias **exactly**, so ANE weights derived from a q8 checkpoint are lossless. Compare its speed with per-channel int8.
6. **Stream ordering:** replace `MFLUX_ANE_ORDER_HACK` with a principled solution. Investigate how MLX schedules CPU and GPU streams and document why the ANE half must be launched first.
7. **W8A8 overflow safety:** confirm where the scales are applied in the Core ML graph, and whether the ANE accumulates in int32 or fp16. Test worst-case partial sums with outlier tokens.

**Go/no-go:** results table plus a recommended packaging layout (enumerated, multifunction or per-tier) before 3b starts.

## Stage 3b: Full DiT integration
- **Coverage:** all 30 main blocks and both noise-refiner blocks; per block, the QKV, o, w1w3 and w2 layers. **Skip** the context refiner (text tokens only, too small), adaLN, the embedders and the final layer.
- **Split ratio per shape,** from 3a, held in config.
- **Weight derivation:**
  - From **bf16:** per-channel int8 (or the 3a format).
  - From **q8:** the lossless group-wise format if 3a passes; otherwise dequantise and requantise, with a logged warning.
- **Optional `--dequantize-on-load`:** for a q8 source, convert the **GPU half** to bf16 at load. It recovers the ~14% q8 penalty for about +3 GB.
- **fp16 range audit:** every split layer, every step, at least 5 prompts, 1024² and 1.5 MP. Per-layer headroom factor (default 32). Any layer whose scaled peak exceeds ~30,000 stays on the GPU.
- **Text-length variation:** row padding to the tier makes caption length irrelevant to ANE shapes.
- **Errors:** if any ANE call fails, raise a clear error rather than silently falling back.

## Stage 3c: Packaging, cache and CLI

**Model folder (artifacts):**
```
Z-Image-Turbo-mflux-q8/
├─ transformer/  text_encoder/  vae/        (existing)
└─ transformer_ane/
     ├─ manifest.json
     └─ ane.mlpackage            (or per-tier files, per the 3a result)
```
`manifest.json` records: model ID and revision, source format and checksum, ANE weight format, precisions included, split ratios, tier token counts, headroom factors, and the coremltools, macOS and chip family it was built with.

**Compiled cache:** `~/.cache/mflux/ane/<key>/…mlmodelc`. The key covers the manifest fields plus the macOS build. It rebuilds automatically when any of them changes.

**CLI:**

| Command or flag | Behaviour |
|---|---|
| `mflux-generate … --ane` | Enable the split (W8A16). Auto-prepares the needed tier on first use, with a progress message |
| `--ane-precision {w8a16,w8a8}` | Precision of the ANE half |
| `--ane-split 0.5` | Advanced override of the GPU share |
| `--dequantize-on-load` | GPU half to bf16 when the source is q8 |
| `--ane-cache-dir PATH` | Override the cache location |
| `mflux-save … -q 8 --ane [--ane-tiers 1k,1.5k] [--ane-precision …]` | Save the model plus `transformer_ane/`. Builds only the 1k tier by default |
| `mflux-ane-prepare --model … --tiers …` | Build packages and compile the cache ahead of time (servers) |
| `mflux-ane-cache list` / `clear` | Inspect and remove compiled caches |

**Gating and messages:**
- Requires macOS 26 or later and an M3 or M4. On M5: warn that the GPU path is likely faster, and allow it with an explicit flag. M1/M2: refuse.
- A q4 or other source with `--ane`: refuse ("ANE mode needs a bf16 or 8-bit model").
- Token count above the largest tier: run that generation on the GPU, with a one-line notice.
- Startup banner, e.g. `ANE: W8A16, 50% of DiT linear layers, tier t4608, cache hit`.
- Timing JSON fields: `ane_enabled`, `ane_precision`, `ane_tier`, `ane_prepare_s`, and per-step times.

**Extension build:**
- A reproducible build script, with nanobind pinned to the version matching the installed MLX.
- Decide and document how it ships: built at install, or a prebuilt wheel. Recommend a build script plus a clear error if it's missing, for now.

## Stage 3d: Benchmark and report
- **Models:** Z-Image Turbo (8 steps) and Base. **Resolutions:** 512², 1024² and ~1.5 MP.
- **Modes:**
  1. bf16 on the GPU
  2. q8 on the GPU
  3. Split W8A16, GPU half q8
  4. Split W8A16, GPU half bf16 (dequantised on load)
  5. Split W8A8, GPU half q8
- **Method:** saved prompt embeddings, fixed seeds, 1 warm-up plus 3 timed runs per configuration. Report step time from step 2 on, total time, first-run prepare time and cache size on disk.
- **Memory:** MLX active + cache memory, **swap change**, and the `aned` process's footprint. Peak alone is not enough.
- **Quality:**
  - An image grid per prompt (bf16, q8 and every split mode) for **Scriv's eyeball review**, at least 6 prompts including a detailed scene
  - PSNR and LPIPS (a perceptual similarity score) against bf16, with **q8 vs bf16 as the yardstick**

## Acceptance criteria
- Split W8A16 with the GPU half in q8 gives **≥1.3× per-step speed against q8** at 1024² on Turbo.
- Memory is no higher than q8 + 10%, counting mflux and `aned` together.
- Scriv accepts the W8A16 and W8A8 images by eye.
- Default behaviour (no `--ane`) is unchanged, and existing tests pass. There are new tests for the split layer, manifest and cache-key logic, and gating messages.
- Reports quote commit hashes and numbers from tool output only.

## Non-goals
- Text encoder, VAE or attention on the ANE.
- M5-specific paths.
- Pre-built ANE packs on Hugging Face (a later stage).
- Models other than Z-Image, though the design should be model-agnostic where cheap.

## Working rules
- Use the `spikes/BENCH.lock` protocol for all benchmarks.
- Ask Scriv before real-weight runs (the memory watchdog applies).
- Work on `experiment/zit-on-ane`; push to the fork only.

## Suggested delegation
- 3a tests and 3b integration: a Sonnet-class agent.
- 3c packaging and CLI: a Sonnet-class agent, after 3a's layout decision.
- Go/no-go reviews at the end of 3a and 3d: an Opus-class agent.
