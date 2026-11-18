# Fused vs separate w1/w3, head-to-head

Date: 2026-11-18, 21:36-21:38. Machine: M4 Max, macOS 27.0.1. Branch `experiment/zit-on-ane-stage3`. Raw evidence: `results/fused_sep.jsonl`. Script: `stage3a/fused_sep.py`. Run under `watchdog.sh` and the shared `BENCH.lock` after the quiet gate passed (21:36:19).

## Method

Three arms, interleaved within each of 8 rounds. T=4224, W8A16 per-channel int8, channels-first 1x1 conv, `--units ane` (CPU_AND_NE), 30 timed samples per round per arm (200 for the tiny arm). One sample of arm b times two back-to-back predictions (`anebench --calls 2`).

| Arm | Model | Calls per sample |
|---|---|---|
| a_fused | c1_4224x20480x3840 | 1 |
| b_separate | c1_4224x10240x3840 | 2 (w1 then w3) |
| c_overhead | c1_64x3840x3840 (tiny) | 1 |

Note on machine state: co-tenant processes were active during the run (a browser burst passed the gate seconds before start). The runner interleaves the arms round by round, so all three see the same conditions. The bimodal ANE state is visible in the round-to-round spread.

## Results

Per-round medians (ms):

| Round | a_fused med | a_fused p90 | b_separate med | b_separate p90 | c_overhead med |
|---|---|---|---|---|---|
| 1 | 55.74 | 99.81 | 78.89 | 98.57 | 0.462 |
| 2 | 82.94 | 95.48 | 85.60 | 101.26 | 0.456 |
| 3 | 54.00 | 58.98 | 81.33 | 91.67 | 0.593 |
| 4 | 57.49 | 61.65 | 65.63 | 89.44 | 0.345 |
| 5 | 68.40 | 82.91 | 75.65 | 95.75 | 0.492 |
| 6 | 59.77 | 86.97 | 85.88 | 92.48 | 0.426 |
| 7 | 55.41 | 61.16 | 63.19 | 84.14 | 0.360 |
| 8 | 55.07 | 89.48 | 77.58 | 86.68 | 0.448 |

Summary:

| Stat | a_fused | b_separate | c_overhead |
|---|---|---|---|
| Best-round median | **54.00 ms** | **63.19 ms** | 0.345 ms |
| Median of round medians | 56.61 ms | 78.24 ms | 0.452 ms |
| Best-round p90 | 58.98 ms | 84.14 ms | 0.45 ms |
| Median of round p90s | 84.94 ms | 92.08 ms | 0.60 ms |
| Throughput at best median | 12.31 TFLOPS | 10.51 TFLOPS (both calls) | — |

Per-call overhead (arm c): the tiny model computes ~0.153 ms of conv work at the fused arm's best-state rate. Its measured call median is 0.345-0.452 ms. That bounds the fixed cost of one Core ML prediction at **~0.19-0.30 ms**, including dispatch, scheduling and the buffer handling in the bench harness.

## Reading

1. **Fused wins in every same-session comparison.** Best state: 54.0 ms against 63.2 ms, so the fused call is 14.5% faster than two separate calls. Median state: 56.6 against 78.2 ms (27.6% faster).
2. **Dispatch overhead does not explain the gap.** Two extra dispatches cost ~0.4-0.6 ms. The observed best-state gap is 9.2 ms. The separate arm also runs at lower throughput per call (10.5 against 12.3 TFLOPS). Back-to-back separate predictions do not reach the fused call's ANE efficiency.
3. The Stage 1 note "2x ~22.8 ms separate beats ~53 ms fused" does not reproduce in this session. Stage 1 measured single-model loops on an idle machine. Tonight the separate arm's best per-call time is 31.6 ms. Under identical conditions the fused call is faster.
4. Fewer calls also cut the per-step in-process cost. 4 ANE calls per block against 7 saves 90 dispatches per step (30 blocks), ~20-30 ms per step at the measured per-call overhead. It also removes MLX join nodes from the graph.

## Recommendation for 3b

**Use the fused forms**: qkv as one 11520x3840 call, w1w3 as one 20480x3840 call, per the Stage 3a layout (4 ANE calls per block). Do not split w1/w3 into separate ANE calls. One uncertainty remains: Stage 1 suggested a single separate call in isolation might win per FLOP. That does not change the decision. Production issues the calls back-to-back per block, which is exactly what arm b measured, and there the fused call wins by 14.5-27.6%.
