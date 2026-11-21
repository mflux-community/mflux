# 3b-2: ANE integration into the Z-Image DiT

Date: 2026-11-19. Worktree `experiment/zit-on-ane-stage3`. Scope: the V0 split
design integrated as `--ane`, plus the checkpoint measurements Scriv asked for
before I polished the integration. All weights here are synthetic (random
transformer). I loaded no checkpoint. 3b-3 to 3b-5 still wait for your approval.

## What I built

- `src/mflux/models/z_image/ane/`
  - `ane_gate.py`: refuses --ane on non-Darwin, macOS below 26, M1/M2 chips,
    q3/q4/q5/q6 sources, --float32 and --compute-precision combos. M5 refuses
    without --ane-allow-m5. `warn_if_background_qos()` warns when the process
    runs at background QoS, because launchd background sessions throttle the
    ANE (Stage 3a: 2.5 W against 9.3 W, about 1.6x slower).
  - `ane_extension.py`: lazy native-module loader (`MFLUX_ANE_EXT_DIR`,
    default the spike build directory).
  - `ane_model_builder.py`: the enumerated W8A16 / W8A8-pre Core ML builders,
    content-hashed cache under `MFLUX_ANE_CACHE` (default
    `~/Library/Caches/mflux/ane`). Only the BUILD path needs coremltools.
  - `ane_layers.py`, `z_image_ane.py`: the V0 split layers and the patcher.
    30 main blocks + 2 noise-refiner blocks, four fused call sites each (qkv,
    o, w1w3, w2 with K-split 4 and headroom 32). Context refiner, adaLN,
    embedders and the final layer stay on the GPU. The GPU suffix keeps the
    source layer's precision: dense bfloat16 rows, or the retained quantized
    rows of a -q 8 model (packed rows slice cleanly).
- CLI: `--ane`, `--ane-precision {w8a16,w8a8}`, `--ane-gpu-share` (default
  0.5), `--ane-cache`, `--dequantize-on-load`, `--ane-allow-m5` on both
  `mflux-generate-z-image` and `mflux-generate-z-image-turbo`. Without --ane
  nothing changes. I updated the pinned CLI contract tests only for the
  new kwargs.

## Two MLX scheduling rules this integration had to learn

`block_q8_sweep.py` bisect (paired, interleaved, this machine, MLX 0.32.4):

1. Drain rule. An ANE primitive whose input chain MLX has not materialised
   stalls the whole graph: the eager W8A8 module cost 54.7 ms against 35.0 ms
   with `mx.eval(q, tok)` before the call. mx.compile forbids an inner eval
   and schedules the chain itself, so compiled mode needs no drain. The
   split layers expose a `drain` flag. The patcher sets it from `compiled`.
2. Layout rule. The GPU half must consume the SHARED PADDED input array and
   the ANE output must be cast full width BEFORE row slicing. Any variant
   that casts a sliced view, or matmuls the unpadded tensor, loses the
   GPU/ANE overlap (32 ms to 55 ms at w13). The earlier "true rows only"
   micro-optimisation was the poison, not a saving.

## Checkpoint 1: block time, W8A8, split-ratio re-sweep

Block 15, T=4128 padded to tier 4608, compiled + order-hack, 6 interleaved
rounds, BENCH.lock + watchdog + macmon, foreground. `3b-2 checkpoint` commit
message and `results/block_q8_sweep.jsonl` hold the raw data. Cosines against
the W8A16 V0 split: 0.99999 across all arms.

| Arm | Median ms | Ratio vs base 157.9 | ANE W mean |
|---|---|---|---|
| w8a16@50 | 124.3 | 0.786 | 1.67 |
| w8a8@50 | 123.6 | 0.781 | 3.02 |
| w8a8@45 | 119.3 | 0.751 | 2.53 |
| w8a8@40 | 116.6 | 0.740 | 2.72 |
| w16@45 | 124.1 | 0.785 | 3.47 |
| w16@40 | 129.2 | 0.816 | 4.13 |

Findings. W8A8 beats W8A16 at every share at block level. More ANE columns
help W8A8 (40:60 best at 116.6 ms, within 2 ms of the ~115 ms target) but
hurt W8A16 (the ANE becomes the critical path). W8A16@50 now reads 124.3 ms
against the assembly report's V0 118.8 ms. The gap is the tier-4608 padding
(+9% rows against the fixed-4224 models) and the extra pad copy. Cross-run
absolute drift applies (same-session base reads 157.9 against 158.8 there).

## Checkpoint 2: whole-step battery, synthetic weights

### A first contaminated pass, and how it was caught

The first battery ran while the machine swapped hard: ~17 GB compressor,
~11 GB swap, ~200 MB free. One 32-block patch measured 80.4 s per step
against a 4.99 s base. Two ladder arms died with
`kIOGPUCommandBufferErrorTimeout`. Part of that was co-tenant load (a
1.8 GB Virtualization VM owned by another agent, plus a second 5 GB python),
and part was mine: I ran a manual reproduction while the battery was live.
Both numbers were thrown out. The battery now gates every timed run on
free memory >= 25% and swap <= 2 GB, and logs peak RSS, free % and swap
before and after each row. You rebooted the machine and ran `mem-free`
before the runs below.

### The clean marginal ladder, and the real blocker

Runs at 1024x1024, compiled mode, `--patch-first N`, sequential under one
lock, rebooted machine, swap 0.4 GB and stable throughout
(`results/step_battery.jsonl` rows pass 1, then the ladder files):

| N blocks patched | step ms | RSS MB | free % before and after |
|---|---|---|---|
| 1 | 5007 | 3719 | 88 -> 80 |
| 2 | 5004 | 4320 | 88 -> 77 |
| 4 | 4949 | 5601 | 88 -> 72 |
| 8 | 5858 | 5079 | 90 -> 28, 4.7 GB swapped during run |
| 16 | 22678 | 6823 | 89 -> 55, 7.0 GB swapped |
| 32 | refused by MEMGATE | - | swap already 7.9 GB |

The base row (memory columns predate the gate) is the clean pre-reboot
pass-1 line: 4988.6 ms, canary 55.5 ms.

A single patched block adds almost nothing and saves almost nothing at step
level. Four blocks give 40 ms. Eight blocks lose 870 ms and drive the
machine into swap. The cause is not the ANE plan cache and not CoreML
packaging. Direct probe (`spikes/ane_matmul/stage3b/probe_model_rss.py`,
`probe_predict_latency.py`, and the vm_stat runs): each live `AneModel`
instance wires roughly 170 MB of non-compressible, non-swappable kernel
memory, a second instance of the same package adds more rather than
sharing, and the wired pages do not return after you free the model.
One patched block holds four models: measured +1000 MB wired per block,
stable across blocks 0-7. Thirty-two patched blocks therefore ask for
about 32 GB of wired memory on a 36 GB machine, before the model itself
(3.5 GB) or the text encoder and VAE that real users keep loaded. That is
why per-predict latency inflates from the clean 6-26 ms measured in
isolation to ~280 ms in the big graph, and why the step collapses.

The block-level 124 ms from Checkpoint 1 is real: it is four predicts per
block, and the sweep process held only four live models. The projection
"~1.3x per block carries to the step" fails for a memory reason, not a
scheduling reason.

### 512x512, reported as measured

512² gives T=1088 rows. The smallest production tier is 4608, so the ANE
computes 4.2x the needed rows at the same weight traffic. Even without the
wired wall, the tier set is wrong for small resolutions. A 1536/2048 tier
is the fix. I completed no 512² timed run, deliberately: the wired
finding above dominates any tier question.

## Verdict and options

3b-2 integration: complete and safe (flags, gate, fail-loud, cache, tests,
default path unchanged). 3b-2 performance: NOT GO on this 36 GB machine.
The --ane split cannot deliver a whole-model win here, because Core ML
wires ~1 GB per patched block and the model needs 32.

What would change the verdict, in order of credibility:

1. Measure on a 64/128 GB M4 Max. The extrapolated best case is the
   block-level 0.79 ratio over the 78% of step time that is linear
   (~1.2x per step), at the cost of ~32 GB wired and 5.7 GB of disk. The
   decision belongs to Scriv with a bigger machine, not to me by
   extrapolation.
2. Cut the live-model count: the wiring scales with instances. Options
   exist (one shared MLModel per distinct shape class with per-call
   weight swaps - Core ML does not expose that. CPU-and-GPU compute units
   for the small call sites - drops the ANE premise) but none is
   demonstrated.
3. Dequantize-free W8A16 packaging does not help: each live model wires its
   own kernel pages, so the wall scales with instances, not weight bytes.

Recommendation: freeze the ANE lane after this report. Keep the branch and
the extension for a large-RAM trial. Spend the next effort on the
Qwen-Image fused QK-norm RoPE port, which has no memory wall.
