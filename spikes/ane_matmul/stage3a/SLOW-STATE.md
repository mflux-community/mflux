# Slow-state probe: why the same ANE work takes ~1.5x longer at random

Date: 2026-11-18, 21:38-21:49. Machine: M4 Max, macOS 27.0.1. Branch `experiment/zit-on-ane-stage3`. Raw evidence: `results/slow_state.jsonl` (summary), `results/slow_state_percall.jsonl` (8387 timestamped samples), `results/slow_state_sys.jsonl` (system samples). Scripts: `stage3a/slow_state_probe.py`, `stage3a/slow_state_sampler.py`. Run under `watchdog.sh` and the shared `BENCH.lock` after the quiet gate passed (21:38:48).

## Method

- Workload: the w1w3 shape (20480x3840, T=4224, W8A16) in a continuous 600-second loop, from `compiled/sa_w1w3.mlmodelc` (coremldata hash `094f7e8eefb125a9`). One prediction per sample, each sample logged with its start timestamp. CPU_AND_NE compute units.
- System sampler: one record every ~2 s with these fields:
  - thermal state (`pmset -g therm`)
  - memory free % (`memory_pressure`)
  - swap in use (`sysctl vm.swapusage`)
  - the sum of all process CPU %, and the top 4 CPU consumers
- ANE power and frequency: this session cannot run `powermetrics`. `sudo -n powermetrics` fails because sudo demands a password, so the probe collected no ANE counters. Scriv can run it manually (command at the end).
- Slow classification: sample >= 70 ms. The distribution is bimodal: fast median 55.6 ms, slow median 82.0 ms, with almost nothing between 60 and 75 ms.

## Results

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

## Correlation: no observable signal predicts the slow state

System conditions joined to the per-sample outcomes (4-second buckets):

| Signal | Fast-sample buckets (n=149) | Slow-sample buckets (n=150) |
|---|---|---|
| Sum of process CPU % | 256.6 | 251.8 |
| Memory free % | 50.95 | 50.91 |
| Swap used MB | 589.5 | 589.5 |
| CPU_Speed_Limit (`pmset -g therm`) | absent (no throttle reported) | absent |
| Top CPU peaks | hindsight-api 100%, duetexpertd 95.8%, app 90.9%, Google 88.2% | identical set |

Fast and slow samples occur under the same CPU load, the same memory state and the same thermal report. The machine was not idle (co-tenant processes were active), but their activity does not separate the two states.

Hypotheses that remain, unconfirmed without ANE counters:

1. ANE-internal clock or QoS states. The ~1.5x ratio with a clean bimodal split resembles a frequency step, matching the GPU clock-state behaviour already documented on this machine.
2. A hidden ANE co-tenant. `duetexpertd` (Apple on-device intelligence) and `hindsight-api` both ran near 100% CPU during the probe, and either could submit ANE work intermittently. ANE usage of other processes is not visible in CPU stats.

Scriv can test hypothesis 1/2 with: `sudo powermetrics --samplers ane_power,thermal -i 1000` during a rerun of `slow_state_probe.py`.

## Proposal: adaptive split

No external cause emerged, so the production path should absorb the state instead of predicting it. Proposal for 3b/3c:

1. Measure both halves in production. The ANE half is easy: record the predict duration inside the bridge (`AneModel::predict`) in a small ring buffer. The GPU half: an MLX event pair around the quantized matmul, or its encoded-kernel timing.
2. Keep an EWMA of ms-per-column for each half, per shape.
3. Pick the split every N layer calls (N=4, one block): choose the variant that minimises max(t_gpu_est, t_ane_est) from a small precompiled set, e.g. ANE shares {40%, 50%, 60%}.
4. Cost: each compiled model bakes in its ANE width, so every variant adds one function per layer. With no weight deduplication (Stage 3a test 4), 3 variants multiply the ANE disk footprint by 3. A cheaper 2-variant set {50%, 33%} covers the slow state at 2x disk. It shifts work to the GPU when the ANE is slow.
5. Fallback if disk matters: keep the static per-shape splits from Stage 3a (they came from best-state numbers) and accept the variance. The split still pays in the slow state. With the 1.47x inflation, the t2-measured w1w3 50:50 ANE half (31.8 ms) becomes the critical path at ~46.7 ms. That is still well below the 65.1 ms q8 solo, so the split needs no static re-balance.

## Consequences for 3b and 3d

- Budget w1w3-class ANE halves at best-state numbers (t2/t4b) but treat p90 as ~1.5x the median in production timing models.
- All 3d benchmark tables need interleaved modes and best-of-N, per the standing rule. A single sequential pass can land one mode in the slow state for its whole run.
- Evaluate the go/no-go criterion (>=1.3x per step) on best-state numbers, with a slow-state sensitivity note. At 1.47x ANE inflation, the w1w3 50:50 layer degrades from 31.8 ms to ~46.7 ms (ANE-bound). That still beats the 65.1 ms q8 solo. The split design degrades gracefully into the slow state.
