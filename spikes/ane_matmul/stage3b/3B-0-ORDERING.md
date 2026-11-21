# 3b-0: ordering inside `mx.compile` — gate PASSED

Date: 2026-11-19, 10:30-10:45. Machine: M4 Max, macOS 27.0.1, foreground session, throttle probe first (46.2 ms on w1w3 fused, tight). Worktree `~/MFlux/mflux-ane-stage3`, branch `experiment/zit-on-ane-stage3`. Code: `stage3b/ordering_block.py`, `stage3b/split_layers.py` (commits `334e6ad7`, `0e41f003`, plus the paired-gate fix). Raw: `stage3b/results/ordering_block.jsonl`, `ordering_block_macmon.jsonl`. Run under `BENCH.lock` and `watchdog.sh` with macmon sampling.

## Method

The harness runs the real Z-Image transformer with random bf16 weights. It reads no checkpoint. It captures block 15's inputs from one real forward: unified stream (1, 4128, 3840), mask, RoPE, t_emb. It replaces the block's four call sites with split modules. Each makes one ANE call over a column prefix: qkv 5760, o 1920, w1w3 10240, w2 1920 with the 4-chunk K-split and headroom 32. GPU halves stay bf16. Six modes run 20 timed calls per round, over 4 interleaved rounds. The modes: base eager, base compiled, split eager hack/nohack, split compiled hack/nohack.

The work found and fixed two driver bugs before trusting this result:

1. The step-level battery could not resolve one patched block (140 ms of a 5.2 s step) and staged assembly in fp32. Replaced by this block-level driver with bf16 assembly.
2. `mx.compile` caches graphs by function object. The first block-level run reused one callable for all three compiled modes, so the "compiled split" numbers showed the unpatched graph. The FAIL verdict of that run was an artifact. Each mode now owns a distinct lambda, traced in its hack state. The rerun below is the fixed one.

## Results (per-round medians, ms)

| Round | base_eager | base_compiled | eager_hack | eager_nohack | compiled_hack | compiled_nohack |
|---|---|---|---|---|---|---|
| 0 | 184.4 | 180.2 | 144.6 | 160.3 | 141.8 | 141.8 |
| 1 | 185.5 | 180.1 | 138.8 | 156.3 | 135.0 | 147.3 |
| 2 | 176.5 | 171.8 | 139.0 | 153.8 | 134.9 | 133.0 |
| 3 | 236.1 | 169.5 | 137.1 | 153.2 | 134.6 | 139.1 |

Round 3 base_eager (236.1 ms) is a disturbed window: the split modes stayed at 133-139 ms and every other round read 176-186 ms. The pairing (median) drops it from the verdict. The row stays visible.

Paired savings (same round, same mode family):

| Round | eager saved | compiled_hack saved | ratio |
|---|---|---|---|
| 0 | 39.8 | 38.4 | 0.965 |
| 1 | 46.7 | 45.1 | 0.966 |
| 2 | 37.5 | 36.9 | 0.984 |
| 3 | 99.0* | 34.9 | 0.353* |

`*` outlier round: an inflated eager base distorts it. The median paired ratio is 0.965. The results file records `gate_paired` with verdict **PASS** (gate: at least 0.9).

ANE activity per mode window (macmon `ane_power`, mean W): base_eager 0.11, base_compiled 0.00, eager_hack 3.03, eager_nohack 3.04, compiled_hack 3.47, compiled_nohack 3.38. The split modes verifiably ran ANE work. The baselines verifiably did not.

## Findings

1. The compiled path overlaps. The compiled split block saves 34.9-45.1 ms against its own compiled base, 96-98% of what the eager split saves. Stage 2's compiled-path serialisation did not reproduce. Two factors likely explain this. This block runs the four fused calls 3b plans to ship, while Stage 2 patched one w2. Also, the compiled sub-block graph may schedule the CPU-stream task before its fused GPU chain. 3b-2 and 3b-5 still run mflux's full-predict graph. This answers the design gate.
2. Eager mode needs the ordering hack. Compiled mode does not. compiled_hack and compiled_nohack sit within noise of each other (133.0-147.3 both), while eager_nohack loses 14-24 ms to eager_hack. Keep the hack enabled everywhere: it costs nothing under compile and it protects every eager path (fallback, capture, debugging).
3. The assembly path still costs real time. The ideal patched block is ~105 ms (t2 bf16 split layer times 69.3 + attention 27 + remainder ~9). The best rounds measured 134.6-145. The extra ~30-40 ms per block comes from pad copies, the fp16 cast, and the 11520/20480-column concats in the current harness. 3b-1 should cut this. Reuse padded buffers and avoid re-casting per call. Test whether the GPU half can write directly into the assembled output.
4. Full-diT projection sanity: one block ratio is 176.5/137.1 = 1.29x. With the assembly overhead halved by 3b-1 work, 1.4-1.5x per block is plausible against the 1.64x ideal. The 3b-5 measurement will decide.

## Validity checklist

- Foreground session, throttle probe first: 46.2 ms, 46.2 ms, 46.2 ms (tight p90). Pass.
- macmon canary: per-mode ANE power separates split from base exactly as expected. No reading fell in the throttled band. Pass.
- Interleaved rounds: four full interleavings. Pass.
- BENCH.lock held by `3b0-block`, watchdog active. Pass.
- All numbers come from `results/ordering_block.jsonl`. Hashes come from git output.
