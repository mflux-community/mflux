# PRD Addendum: Stage 3b, Full DiT Integration (Z-Image ANE Split)

*Supersedes the Stage 3b section of `PRD-stage3.md` wherever they differ. Inputs: Stage 3a results (`STAGE3A-REPORT.md`, `stage3a/*.md`) as of commit `e99ce647`.*

## Goal
Split every large linear layer of Z-Image's DiT between the GPU (MLX) and the ANE (Core ML, int8) **inside mflux**, behind an opt-in flag, and prove the per-step speed-up and image quality on real weights.

**Target:** ≥1.3× per step against q8 at 1024² on Turbo (the PRD target). **Projection:** ~1.64× (Stage 3a, unthrottled).

## Decisions carried in from Stage 3a

| Topic | Decision | Source |
|---|---|---|
| Layer layout | **Fused calls, 4 ANE calls per block:** QKV (3840→11520), o (3840²), w1w3 (3840→20480), w2 (10240→3840, 4×2560 K-split, one baked constant per chunk) | `FUSED-VS-SEPARATE.md`: fused 14–28% faster |
| Op form | 1×1 `Conv2d`, channels-first `(1, C, 1, S)`, baked int8 weights + per-channel scales (W8A16) | Stage 1–3a |
| Split ratio | **50:50 for every shape** (w1w3 with a q8 GPU half: 45:55 optional, within noise) | `T2-UNTHROTTLED.md` |
| Stream ordering | **Keep the ordering workaround** (the ANE half launches first via a data dependency). `async_eval` never wins; no workaround serialises | `T6-UNTHROTTLED.md` |
| Packaging | Multifunction and standalone perform the same. Pick by artifact management; **enumerated shapes work**, so prefer one package with enumerated token counts | Stage 3a |
| Weight derivation | **Per-channel int8 from bf16.** Group-wise int8 matching MLX q8 **failed** in 3a, so from a q8 source, dequantise then requantise per-channel, with a logged warning | Stage 3a |
| fp16 headroom | Divide by **32** before the ANE and multiply after, at minimum on w2. Set per layer from the range audit | Stage 2 |
| Precision | **W8A16 default; W8A8 a first-class option** (`--ane-precision w8a8`), judged by Scriv's eye | Decision 2026-11-18 |
| Adaptive split | **Optional insurance only, not built in 3b.** The "slow state" was launchd `ProcessType=Background` throttling, not the hardware | `SLOW-STATE.md` run 3 |

## Benchmark validity rules (mandatory)
1. **Run in a foreground session,** never from a Background-QoS process. Run the throttle probe first: w1w3 at T=4224 must give a median of ~46 ms with a tight p90. If it doesn't, stop.
2. **Use macmon as a canary:** log the ANE rail power for every run. Runs that read in the throttled band (~2.5–5 W sustained, with matching latency inflation) are invalid.
3. **Interleave the arms;** no sequential best-of blocks.
4. **Use `BENCH.lock` and `watchdog.sh`,** as before.
5. **Keep ops scripts in `stage3a/ops/`** (or `stage3b/ops/`), never only in `/tmp`.
6. **Hashes and numbers come from tool output only.**

## Work items, in order

### 3b-0: Ordering inside `mx.compile` (first, it can block the design)
Stage 2 saw the two halves run one after the other inside mflux's compiled `predict`. Using the real integration harness, run in the foreground after the throttle probe:
- Patch one block's four layers.
- Measure the step's linear-layer time with `mx.compile` on and off, and with and without the ordering workaround.
- **Gate:** the compiled path must overlap at least as well as the non-compiled path (within 10%). If it doesn't, either keep the split layers outside the compiled region (and measure that cost), or find an ordering that survives compilation.

### 3b-1: The `ane_linear` extension, production-ready
- Variants: **fused QKV, o, w1w3, and w2 with K-split**, in W8A16 and W8A8.
- Enumerated token counts: tiers 4608 / 6528 / 8320 / 16128, with **row padding** (pad the ANE input rows up to the tier and slice the output). Attention keeps its true length.
- Fail loudly: any Core ML prediction error raises; no silent fallback.
- A reproducible build, with nanobind pinned to the installed MLX version.

### 3b-2: Integration into the DiT
- **Coverage:** 30 main blocks plus 2 noise-refiner blocks, 4 calls each. Skip the context refiner, adaLN, the embedders and the final layer.
- **GPU half:** MLX bf16, or q8 via `-q 8`. Add `--dequantize-on-load` to turn a q8 GPU half into bf16 at load (~+3 GB).
- **Weights:** at load, split each layer by columns. The GPU keeps its columns; the ANE columns come from the cached Core ML packages and are dropped from MLX.
- **Flag:** `--ane` (W8A16) and `--ane-precision {w8a16,w8a8}`. Default behaviour is unchanged.
- **Gating:** macOS 26+, M3/M4 only; on M5, warn and require an explicit flag; refuse on M1/M2; refuse q4 and other sources ("ANE mode needs a bf16 or 8-bit model").
- **Background-QoS warning:** at start-up, if the process runs at Background QoS, print a warning that ANE throughput will be reduced.

### 3b-3: fp16 range audit (real weights; ask Scriv first)
- Every split layer, every step, ≥5 prompts, at 1024² and ~1.5 MP, Turbo and Base.
- Record the peak |x| before and after headroom scaling. Set the headroom per layer; a layer whose scaled peak exceeds ~30,000 stays on the GPU.
- For W8A8: also record the worst-case partial sums, and confirm where the scales are applied.

### 3b-4: Accuracy (real weights; ask Scriv first)
- Per-layer cosine against fp32: W8A16 > 0.9999, W8A8 > 0.999.
- Whole-image PSNR and LPIPS against bf16, with **q8 vs bf16 as the yardstick**, same seeds.
- **Image grid** for Scriv's eyeball review: bf16 / q8 / split W8A16 / split W8A8, ≥6 prompts including a detailed scene, at 1024² and 1.5 MP.

### 3b-5: End-to-end benchmark (foreground, canary on)
- Turbo (8 steps) and Base. 512², 1024² and ~1.5 MP.
- Modes: bf16, q8, split W8A16 with q8 GPU half, split W8A16 with bf16 GPU half, split W8A8 with q8 GPU half.
- Use saved prompt embeddings. Report step time from step 2 onwards, total time, first-run preparation time and cache size.
- Memory: MLX active and cache, **swap delta**, and the `aned` process footprint.

## Acceptance criteria
- **3b-0 gate passed,** or an alternative with measured cost.
- **Split W8A16 with a q8 GPU half: ≥1.3× per step against q8** at 1024² on Turbo (projection ~1.64×).
- Memory ≤ q8 + 10%, counting mflux and `aned` together.
- W8A16 and W8A8 images accepted by Scriv by eye.
- Every benchmark run passed the throttle probe and the canary.
- Default behaviour (no `--ane`) unchanged; existing tests pass; new tests cover the split layer, padding, gating and errors.

## Out of scope (Stage 3c and later)
- `mflux-save --ane`, `mflux-ane-prepare`, the cache commands and the `transformer_ane/` manifest.
- The adaptive split controller.
- Pre-built packs on Hugging Face.
- Text encoder, VAE or attention on the ANE.

## Working rules
- Branch `experiment/zit-on-ane-stage3`, worktree `~/MFlux/mflux-ane-stage3`.
- **Ask Scriv before** any real-weight run (3b-3, 3b-4, 3b-5), any push, or anything outside the worktree.
- STE lint on every written report. Stop and report after each work item.
