# Slow-state probe: why the same ANE work takes ~1.5x longer at random

Date: 2026-11-18. Run 1: 21:38-21:49. Run 2 (with `macmon` ANE power monitoring): 22:09-22:20. Machine: M4 Max, macOS 27.0.1. Branch `experiment/zit-on-ane-stage3`.

Raw evidence: `results/slow_state.jsonl` (summaries, both runs), `results/slow_state_percall*.jsonl` (timestamped samples), `results/slow_state_sys*.jsonl` (system samples), `results/slow_state_macmon_run2.jsonl` (ANE/GPU power at 1 s). Scripts: `stage3a/slow_state_probe.py`, `stage3a/slow_state_sampler.py`. Both runs passed the quiet gate and ran under `watchdog.sh` and the shared `BENCH.lock`.

**Headline (run 2):** the slow state shows up directly in ANE power. Fast buckets draw **5.01 W** from the ANE rail, slow buckets **2.50 W** — half the power for the same work per prediction, while GPU frequency stays flat (777 against 773 MHz). That is the signature of a downclocked or QoS-limited ANE, not of a co-tenant stealing ANE time (a co-tenant would raise total ANE power, not halve it).

## Method

- Workload: the w1w3 shape (20480x3840, T=4224, W8A16) in a continuous 600-second loop, from `compiled/sa_w1w3.mlmodelc` (coremldata hash `094f7e8eefb125a9`). One prediction per sample, each sample logged with its start timestamp. CPU_AND_NE compute units.
- System sampler: one record every ~2 s with these fields:
  - thermal state (`pmset -g therm`)
  - memory free % (`memory_pressure`)
  - swap in use (`sysctl vm.swapusage`)
  - the sum of all process CPU %, and the top 4 CPU consumers
- ANE power and frequency: run 1 collected no ANE counters. `sudo -n powermetrics` fails because sudo demands a password. Run 2 instead logged the ANE power rail with `macmon pipe -i 1000`, a sudoless IOReport reader (macmon 0.12.0).
- Slow classification: sample >= 70 ms. The distribution is bimodal: fast median 55.6 ms, slow median 82.0 ms, with almost nothing between 60 and 75 ms.

## Run 1 results

| Metric | Value |
|---|---|
| Samples | 8387 in 600 s (median gap 73.2 ms) |
| Slow samples | 4538 (54.1%) |
| Fast-state median | 55.61 ms (12.0 TFLOPS) |
| Slow-state median | 82.03 ms (8.1 TFLOPS) |
| Slow/fast ratio | 1.47x |
| Whole-loop median / p90 | 73.14 / 87.58 ms |
| Longest slow runs | 5.1, 5.0, 4.2, 4.1, 4.0 s (top 5) |

The state flaps on the scale of seconds. The longest slow run is ~5 s, and slow time is spread over ~90 runs. Neither state is stable for minutes during this window.

## Run 1 correlation: no system signal predicts the slow state

System conditions joined to the per-sample outcomes (4-second buckets):

| Signal | Fast-sample buckets (n=149) | Slow-sample buckets (n=150) |
|---|---|---|
| Sum of process CPU % | 256.6 | 251.8 |
| Memory free % | 50.95 | 50.91 |
| Swap used MB | 589.5 | 589.5 |
| CPU_Speed_Limit (`pmset -g therm`) | absent (no throttle reported) | absent |
| Top CPU peaks | hindsight-api 100%, duetexpertd 95.8%, app 90.9%, Google 88.2% | identical set |

Fast and slow samples occur under the same CPU load, the same memory state and the same thermal report. The machine was not idle (co-tenant processes were active), but their activity does not separate the two states.

## Run 2 (with macmon): ANE power separates the states

Same workload and method, 22:09-22:20, plus `macmon pipe -i 1000` (sudoless IOReport reader) logging ANE power, GPU frequency and activity at 1 s. 6773 samples.

Distribution (run 2 against run 1):

| Metric | Run 1 | Run 2 |
|---|---|---|
| Slow fraction | 54.1% | 69.7% |
| Fast-state median | 55.61 ms | 66.73 ms |
| Slow-state median | 82.03 ms | 108.80 ms |
| Slow/fast ratio | 1.47x | 1.63x |
| Longest slow runs | ~5.1 s | ~7.9 s |
| Whole-loop median / p90 | 73.1 / 87.6 ms | 79.7 / 114.0 ms |

Run 2 ran in a lower ANE regime overall: even its fast state (66.7 ms) was slower than run 1's fast state (55.6 ms). The two states exist inside each run, and the whole level also drifts between runs.

macmon join — per-call samples bucketed by slow fraction into the overlapping 1-second macmon records:

| Signal | Fast buckets (n=34) | Slow buckets (n=352) |
|---|---|---|
| **ANE power (W)** | **5.01** | **2.50** |
| GPU frequency (MHz) | 777.0 | 772.5 |
| GPU active ratio | 0.178 | 0.073 |
| CPU active ratio | 0.163 | 0.105 |
| Total chip power (W) | 15.79 | 6.81 |

The sys-sampler join again showed no separation (CPU busy 135 against 120%, memory free ~85% both, swap ~1582 MB both).

## Conclusion

The slow state is an ANE clock/QoS state. Evidence:

1. ANE power halves (5.01 to 2.50 W) while the work per prediction stays identical. Latency rises 1.6x at the same time.
2. GPU frequency does not move (777 against 773 MHz), so this is not a whole-chip power cap.
3. A hidden ANE co-tenant would raise total ANE power during slow samples. The measured power falls, which rules out contention as the direct cause. Apple-intelligence daemons (`duetexpertd`, `siriactionsd`, `ANECompilerService`) were active in both fast and slow buckets alike.
4. The behaviour matches the two-clock-state pattern already documented for short GPU kernels on this machine.

Run 2 resolves the earlier hypotheses list in favour of hypothesis 1 (ANE-internal clock/QoS states). `powermetrics` stays unavailable to agents (`sudo -n` demands a password). `macmon` provides the ANE power rail without sudo, and it answered the question.

## Proposal: adaptive split

The production path should absorb the state. Proposal for 3b/3c:

1. Detect the state directly: read the ANE power rail the way `macmon` does (IOReport, no sudo), or simply measure achieved ms-per-column. Record the predict duration inside the bridge (`AneModel::predict`) in a small ring buffer. Time the GPU half with an MLX event pair around the quantized matmul.
2. Keep an EWMA of ms-per-column for each half, per shape.
3. Pick the split every N layer calls (N=4, one block): choose the variant that minimises max(t_gpu_est, t_ane_est) from a small precompiled set, e.g. ANE shares {40%, 50%, 60%}.
4. Cost: each compiled model bakes in its ANE width, so every variant adds one function per layer. With no weight deduplication (Stage 3a test 4), 3 variants multiply the ANE disk footprint by 3. A cheaper 2-variant set {50%, 33%} covers the slow state at 2x disk. It shifts work to the GPU when the ANE is slow.
5. Fallback if disk matters: keep the static per-shape splits from Stage 3a (they came from best-state numbers) and accept the variance. The split still pays in the slow state. With run 2's 1.63x inflation, the t2-measured w1w3 50:50 ANE half (31.8 ms) becomes the critical path at ~51.8 ms. That is still below the 65.1 ms q8 solo, so the split needs no static re-balance.

## Consequences for 3b and 3d

- Budget w1w3-class ANE halves at best-state numbers (t2/t4b). Treat p90 as ~1.6x the median in production timing models. The whole level also drifts between sessions: run 2's fast state matches run 1's mid range.
- All 3d benchmark tables need interleaved modes and best-of-N, per the standing rule. A single sequential pass can land one mode in the slow state for its whole run.
- Evaluate the go/no-go criterion (>=1.3x per step) on best-state numbers, with a slow-state sensitivity note. At run 2's 1.63x ANE inflation, the w1w3 50:50 layer degrades from 31.8 ms to ~51.8 ms (ANE-bound). That still beats the 65.1 ms q8 solo. The split design degrades gracefully into the slow state.
