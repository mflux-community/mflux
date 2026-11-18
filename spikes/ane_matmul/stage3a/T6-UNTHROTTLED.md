# t6 re-run: stream ordering without session throttling

Date: 2026-11-19, ~10:40. The method matches Stage 3a test 6: synthetic w2 (3840x10240), 4128 tokens padded to 4224, 50:50 split, GPU half q8, headroom 1/32, eager mode. Rounds alternate the strategies, best of 3, under `BENCH.lock` and `watchdog.sh`. The session passed the throttle probe first: six passes, all 45.8-46.2 ms on the w1w3 model. Raw evidence: `results/t6_ordering.jsonl` (this run) and `results/t6_ordering_throttled.jsonl` (Stage 3a evening run).

## Results

| Strategy | Throttled (ms) | Throttled vs GPU-only | Unthrottled (ms) | Unthrottled vs GPU-only |
|---|---|---|---|---|
| GPU-only q8 anchor | 31.0 | 1.00x | 30.8 | 1.00x |
| none | 38.9 | 0.80x | 34.8 | 0.89x |
| hack | 24.3 | 1.27x | **19.5** | **1.58x** |
| `mx.async_eval` | 46.8 | 0.66x | 34.6 | 0.89x |

Correctness unchanged: all three strategies produce bitwise identical outputs (max abs diff 0.0).

## Reading

1. The in-process split pays 1.58x on w2 when unthrottled. The windowed t2 prediction for w2 50:50 is 1.86x (17.0 ms). The gap is now 2.5 ms of join overhead, not the 4-7 ms the throttled data suggested.
2. 19.5 ms matches the Stage 2 real-weight in-mflux measurement (21.3 ms at 4128 tokens under whatever state the evening before showed). The spike's in-process path is consistent across sessions.
3. `none` still serialises (0.89x against 0.80x throttled). The ordering dependency remains mandatory.
4. `async_eval` is neutral-to-harmful (0.89x, same as none, and 0.66x under throttling). It never wins. Design rule 6 from Stage 3a stands unchanged: keep the hack, do not adopt `mx.async_eval`.

## Consequence for 3b

The main risk flagged in `T2-UNTHROTTLED.md` (in-process join overhead eroding the projected step gain) shrinks. w2 runs 1.58x in-process against 1.86x windowed. t2 measured the other three layers windowed only. The ~1.64x per-step projection keeps, with roughly 0.1-0.15x of it still exposed to integration overhead. The remaining open item is `mx.compile`: Stage 2 saw serialisation there even with the hack in place. Re-run this test inside the compiled predict path early in 3b, in a foreground session.
