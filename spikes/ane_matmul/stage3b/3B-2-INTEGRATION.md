# 3b-2: ANE integration into the Z-Image DiT

Date: 2026-11-19. Worktree `experiment/zit-on-ane-stage3`. Scope: the V0 split
design integrated as `--ane`, plus the checkpoint measurements Scriv asked for
before integration was polished. All weights here are synthetic (random
transformer). No checkpoint was loaded. 3b-3 to 3b-5 still wait for approval.

## What was built

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
    `~/Library/Caches/mflux/ane`). coremltools is needed only to BUILD.
  - `ane_layers.py`, `z_image_ane.py`: the V0 split layers and the patcher.
    30 main blocks + 2 noise-refiner blocks, four fused call sites each (qkv,
    o, w1w3, w2 with K-split 4 and headroom 32). Context refiner, adaLN,
    embedders and the final layer stay on the GPU. The GPU suffix keeps the
    source layer's precision: dense bfloat16 rows, or the retained quantized
    rows of a -q 8 model (packed rows slice cleanly).
- CLI: `--ane`, `--ane-precision {w8a16,w8a8}`, `--ane-gpu-share` (default
  0.5), `--ane-cache`, `--dequantize-on-load`, `--ane-allow-m5` on both
  `mflux-generate-z-image` and `mflux-generate-z-image-turbo`. Without --ane
  nothing changes; the pinned CLI contract tests were updated only for the
  new kwargs.

## Two MLX scheduling rules this integration had to learn

`block_q8_sweep.py` bisect (paired, interleaved, this machine, MLX 0.32.4):

1. Drain rule. An ANE primitive whose input chain has not been materialised
   stalls the whole graph: the eager W8A8 module cost 54.7 ms against 35.0 ms
   with `mx.eval(q, tok)` before the call. mx.compile forbids an inner eval
   and schedules the chain itself, so compiled mode needs no drain. The
   split layers expose a `drain` flag; the patcher sets it from `compiled`.
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
against the assembly report's V0 118.8 ms; the gap is the tier-4608 padding
(+9% rows against the fixed-4224 models) and the extra pad copy. Cross-run
absolute drift applies (same-session base reads 157.9 against 158.8 there).

## Checkpoint 2: whole-step battery, synthetic weights

(Numbers fill in when `results/step_battery.jsonl` completes: base, ane16 and
ane8 at 1024x1024 and 512x512, three interleaved passes as alternating
subprocesses under one lock. Patch cost: building 128 Core ML packages takes
about 130 s per precision on a cold cache; loading a warm cache adds about 30 s.)
