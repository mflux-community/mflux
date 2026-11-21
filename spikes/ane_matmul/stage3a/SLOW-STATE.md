# Slow-state probe: why the same ANE work takes ~1.5x longer at random

Date: 2026-11-18. Run 1: 21:38-21:49. Run 2 (with `macmon` ANE power monitoring): 22:09-22:20. Run 3 (from the foreground `qwen serve` session): 22:37-22:48. Machine: M4 Max, macOS 27.0.1. Branch `experiment/zit-on-ane-stage3`.

Raw evidence: `results/slow_state.jsonl` (summaries, all runs). Timestamped samples: `results/slow_state_percall*.jsonl`. System samples: `results/slow_state_sys*.jsonl`. ANE/GPU power at 1 s: `results/slow_state_macmon_run2.jsonl` and `results/slow_state_macmon_run3.jsonl`. Scripts: `stage3a/slow_state_probe.py` and `stage3a/slow_state_sampler.py`. All runs passed the quiet gate and ran under `watchdog.sh` and the shared `BENCH.lock`.

**Headline (run 3): the slow state is gone. The cause is session QoS.** Scriv suspected that the launchd-managed `qwen serve` daemon ran throttled. He restarted it in the foreground. The same probe then produced 13,069 samples, and zero of them were slow. Median 45.89 ms. p90 46.44 ms. Throughput 14.48 TFLOPS. Run 2, from the launchd session, was 69.7% slow with a 108.8 ms slow median. ANE power tells the story. Run 2: 5.01 W in fast buckets, 2.50 W in slow buckets. Run 3: a steady **9.32 W**. ANE work submitted from the background launchd session runs downclocked. The same work from the foreground session runs at full rate.

**Run 2 headline (still valid within its session):** the slow state shows up directly in ANE power. Fast buckets draw **5.01 W** from the ANE rail. Slow buckets draw **2.50 W**. The same work per prediction then takes 1.6x longer. GPU frequency stays flat (777 against 773 MHz). A co-tenant stealing ANE time would raise total ANE power, not halve it.

## Method

- Workload: the w1w3 shape (20480x3840, T=4224, W8A16) in a continuous 600-second loop, from `compiled/sa_w1w3.mlmodelc` (coremldata hash `094f7e8eefb125a9`). One prediction per sample. The logger attaches a wall-clock timestamp to each sample. CPU_AND_NE compute units.
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

## Run 3 (foreground session): the slow state disappears

Identical workload, method and gate. The only change: `qwen serve` runs in the foreground from a terminal. Every child process (the probe, `anebench`, Core ML) then inherits the terminal's foreground QoS instead of the launchd service's background QoS. 13,069 samples in 600 s.

| Metric | Run 1 (launchd) | Run 2 (launchd) | Run 3 (foreground) |
|---|---|---|---|
| Slow fraction | 54.1% | 69.7% | **0.0%** |
| Median | 73.1 ms | 79.7 ms | **45.9 ms** |
| p90 | 87.6 ms | 114.0 ms | **46.4 ms** |
| Fastest state median | 55.6 ms | 66.7 ms | **45.9 ms** |
| Throughput | — | — | **14.48 TFLOPS** |
| ANE power (macmon) | not logged | 5.01 W fast / 2.50 W slow | **9.32 W steady** |
| GPU frequency | — | 777 / 773 MHz | 781 MHz |

- The run-3 latency distribution is tight: p25-p75 spans 45.7-46.2 ms, max 55.7 ms. No second mode exists.
- 45.9 ms matches the Stage 1 idle-machine rate for this shape (2 x 22.8 ms separate w1 calls = 45.6 ms). Every measurement since Stage 1, including the Stage 3a battery, ran degraded to some degree by the session QoS.
- Co-tenant CPU load during run 3 exceeded runs 1-2. One Electron process hit 259.9% and `duetexpertd` hit 98.6%. The slow state never appeared. CPU contention is definitively not the cause.
- The added `anebench` priority logging returned no values (the `pgrep`/`ps` probe missed the process), so the QoS class itself was not captured directly. The session-level A/B is the evidence.

## Conclusion

The slow state is a QoS-driven ANE clock state, tied to the calling process tree:

1. Launchd-managed session (runs 1-2): the ANE rail draws 2.5-5.0 W, and latency flips between two states on the scale of seconds.
2. Foreground user session (run 3): the ANE rail holds 9.3 W, and latency is stable at the hardware's full rate.
3. GPU frequency stays flat across all states (~773-781 MHz), so this is ANE-specific, not a whole-chip power cap.
4. Co-tenant CPU load does not explain it: run 3 had the heaviest CPU co-tenants and zero slow samples. A co-tenant on the ANE itself would raise ANE power, not halve it.

Runs 1-2 answered "what" (a downclocked ANE). Run 3 answers "why": the benchmark processes inherited a throttled QoS from the launchd-managed agent daemon. `powermetrics` stays unavailable to agents (`sudo -n` demands a password). `macmon` provides the ANE power rail without sudo, and it carried the diagnosis.

## Proposal after run 3

The primary fix is operational, not algorithmic:

1. Run every mflux benchmark (3d) and every agent-driven ANE measurement from a foreground terminal session. For agent daemons under launchd, check the service's QoS configuration or run benchmarks through a foreground wrapper.
2. Keep `macmon` ANE power in the benchmark harness as a cheap validity canary: ~9 W on this workload means full rate, ~2.5-5 W means throttled. Flag any timed run below a power floor.
3. The adaptive-split controller (EWMA of ms-per-column per half, re-pick the ANE width variant every block) becomes optional insurance rather than a requirement. End users run mflux from a terminal, which is a foreground session, so they should not meet the slow state. If mflux ever ships a daemon/server mode, revisit it. The design from the run-2 proposal stays valid as written.
4. Fallback numbers still hold: even at run 2's 1.63x inflation the w1w3 50:50 split (~51.8 ms ANE-bound) beat the 65.1 ms q8 solo.

## Consequences for 3b and 3d

- Re-baseline in the foreground session before 3b integration work. The true best-state numbers beat the Stage 3a tables: w1w3 fused runs 45.9 ms (14.5 TFLOPS) against the 52.8-54.0 ms that throttling produced. The t2/t6 split speedups come from one throttled session, so the GPU:ANE balance may shift. Re-run the t2 sweep from the foreground session before freezing the ratios in config.
- All Stage 3a absolute ANE numbers are conservative. The relative conclusions (fusion wins, enumerated works, multifunction parity, group-wise fails) come from single-session comparisons and stand.
- All 3d benchmark tables need interleaved modes, best-of-N, and a macmon ANE-power canary per the standing rules.
- The go/no-go criterion (>=1.3x per step) looks safer than the run-1/2 data suggested. Throttling slowed the ANE half during every Stage 3a measurement and the GPU half less.
