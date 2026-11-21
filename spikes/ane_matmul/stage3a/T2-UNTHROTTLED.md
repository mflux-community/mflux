# t2 re-baseline: concurrency splits without session throttling

Date: 2026-11-19, 09:36-10:12. Machine: M4 Max, macOS 27.0.1, branch `experiment/zit-on-ane-stage3`. Runner: `stage3a/ops/t2_unthrottled.sh` (throttle probe, then quiet gate, then the original `t2_splits.py`). Raw evidence: `results/t2_splits.jsonl` (this run, 48 points) and `results/t2_splits_throttled.jsonl` (the Stage 3a run of 2026-11-18 evening, kept for comparison). macmon canary: `results/t2_macmon.jsonl`.

This re-baseline executes the action item from `SLOW-STATE.md` run 3: freeze the 3b split ratios only on unthrottled numbers. The session passed the throttle probe at 46.2 ms on the w1w3 fused model, which is full rate. macmon held 6.8 W median and 9.8 W peak on the ANE rail across the sweep. The method matches Stage 3a: same models, same window protocol, best of 3 per point, GPU and ANE halves from the same window, `BENCH.lock` and `watchdog.sh`.

## Layer times by split (unthrottled, median of 3 windows)

| Shape | GPU mode | GPU solo | ANE alone | 60:40 | 50:50 | 45:55 | 40:60 | Best |
|---|---|---|---|---|---|---|---|---|
| o 3840² | q8 | 11.8 ms | 9.2 ms | 7.3 | **6.9** | 7.5 | 7.6 | 50:50 = 1.71x |
| o 3840² | bf16 | 10.6 ms | 9.0 ms | **6.7** | 6.8 | 7.5 | 7.6 | 60:40 = 1.60x |
| qkv 11520x3840 | q8 | 34.9 ms | 26.4 ms | 21.2 | **17.8** | 17.9 | 19.3 | 50:50 = 1.96x |
| qkv 11520x3840 | bf16 | 31.5 ms | 26.4 ms | 19.2 | **16.5** | 17.8 | 19.1 | 50:50 = 1.92x |
| w1w3 20480x3840 | q8 | 62.2 ms | 46.3 ms | 37.5 | 31.3 | 30.1 | **29.8** | 40:60 = 2.09x |
| w1w3 20480x3840 | bf16 | 56.6 ms | 46.3 ms | 34.0 | **28.4** | 30.1 | 32.1 | 50:50 = 2.00x |
| w2 3840x10240 | q8 | 31.5 ms | 25.7 ms | 19.2 | **17.0** | 19.1 | 19.3 | 50:50 = 1.86x |
| w2 3840x10240 | bf16 | 28.7 ms | 25.5 ms | 17.5 | **17.2** | 19.1 | 19.3 | 50:50 = 1.67x |

Comparison against the throttled Stage 3a sweep (best speedup per shape/mode):

| Shape / mode | Throttled | Unthrottled | Change |
|---|---|---|---|
| o q8 | 1.31x at 50:50 | 1.71x at 50:50 | ratio holds, speedup improves |
| o bf16 | 1.41x at 50:50 | 1.60x at 60:40 | near-flat, pick 50:50 for config simplicity |
| qkv q8 | 1.98x at 50:50 | 1.96x at 50:50 | unchanged (GPU-bound) |
| qkv bf16 | 1.65x at 60:40 | 1.92x at 50:50 | ratio moves to 50:50 |
| w1w3 q8 | 2.04x at 50:50 | 2.09x at 40:60 | slight ANE-heavy preference |
| w1w3 bf16 | 1.84x at 50:50 | 2.00x at 50:50 | ratio holds |
| w2 q8 | 1.62x at 60:40 | 1.86x at 50:50 | ratio moves to 50:50 |
| w2 bf16 | 1.63x at 60:40 | 1.67x at 50:50 | ratio moves to 50:50 |

## Updated 3b split-ratio recommendation

This supersedes design rule 2 in `STAGE3A-REPORT.md`. The report stays unedited. Cite this file instead.

- **Use 50:50 for o, qkv, w2, and w1w3 with the bf16 half.**
- For w1w3 with the q8 half, 45:55 gains about 5% at best state. The 40:60 and 45:55 points sit within noise of each other.
- The unthrottled ANE is fast enough to take half the columns for every shape. The earlier GPU-heavy ratios came from throttling.

## Updated projection

Use the GPU half in q8 at the best per-shape splits. The per-block linear time drops from 140.4 ms (solo) to 71.4 ms (split), a **1.96x gain on linear layers**. Linear layers take 79-80% of block time, so the projected per-step gain is **~1.64x against q8 alone**. The previous throttled estimate was 1.56x, and the PRD target is 1.3x.

## Caveats

1. These are free-running window numbers. The in-process join carries the costs that `SLOW-STATE.md` and t6 describe. The t6 ordering test also ran throttled. Re-run it in the foreground before 3b freezes the integration design. That is a quick follow-up under the same session rules.
2. Single-session numbers, one evening. The gate passed and the canary held, but 3d should repeat under its own canary checks.
3. bf16 and q8 GPU halves land within a few percent at the recommended 50:50. The default GPU half stays q8 (`--dequantize-on-load` remains optional).
