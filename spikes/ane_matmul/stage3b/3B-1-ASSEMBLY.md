# 3b-1 assembly experiments: column concat beats partial-sum splits

Date: 2026-11-19, ~11:30. Driver: `stage3b/run_variants.py`, variants in `stage3b/variants.py`, raw: `stage3b/results/variants.jsonl` + `variants_macmon.jsonl`. Foreground session, fresh throttle probe (46.0 ms), watchdog + BENCH.lock, compiled+hack mode, 4 interleaved rounds of 20 calls, macmon windows per mode. Correctness: eager block output cosine against V0, target > 0.9999.

## Question

Scriv's priority list: can partial-sum splits beat the column concatenation of V0? Target was block <= ~115 ms.

| Variant | Idea | ANE calls |
|---|---|---|
| V0 | column prefix + concat assembly | 4 (qkv, o, w13, w2-k4) |
| E2 | attention split by heads. Two SDPA calls. Output projection split by input rows, partial sums added | 1 (qkv prefix, head-aligned) |
| E1 | FF split by hidden units (Megatron). w2 split by input rows, partial sums added | 2 chained (w1a and w3a, then w2a) |

## Results

| Mode | Block (ms) | Saved vs base | ANE activity (macmon mean W) | Cosine vs V0 |
|---|---|---|---|---|
| base (unpatched, compiled) | 158.8 | — | 0.17 | — |
| V0 | **118.8** | **40.0** | 3.57 | 1.0 |
| E2 | 122.3 | 36.5 | 3.65 | 0.9999984 |
| E1 | 140.2 | 18.6 | 3.33 | 0.9999980 |
| E1+E2 | 143.3 | 15.5 | 3.04 | 0.9999973 |

Round spread was <= 0.4 ms in every mode. No contention in this window.

## Findings

1. V0 wins. Keep column prefix + concat. E2 is 3.5 ms worse. Removing the concat saves little. The split SDPA costs more than the assembly it removes. E1 is 21 ms worse because its two ANE calls chain: w1a and w3a must finish, SwiGLU runs, then w2a starts. The FF critical path becomes a serial ANE pair. Partial sums only pay when they shorten the critical path. Here they lengthen it.
2. The "30-40 ms assembly overhead" from 3b-0 was mostly misattribution. In this run base reads 158.8 ms. In the 3b-0 run it read 169.5-180.2 ms. Absolute levels drift with machine state. Within one session, V0 sits ~4 ms from the 115 ms target. The remaining candidates attack copies worth a few ms at most. V0 casts once per call already, and the MLX allocator reuses same-shape buffers. Output backings were already in place since Stage 2: the bridge sets them and the ANE writes into the MLX buffer.
3. Cross-session absolutes are not comparable. Base reads 158.8 ms here against 169.5-180.2 ms in 3b-0. Every verdict in this document is same-session and paired for that reason.
4. The E1/E2 code stays in `variants.py` as reference. If a future shape makes the ane-prefix call much longer than the GPU half, E2-style head splitting becomes the fallback. It removes the mid-graph wait on the qkv result. Not today.

## Decision

**3b-1 assembly decision: keep V0 unchanged.** The existing design meets the goal within 4 ms of target. Both proposed alternatives are measurably worse. Move on to the production items: enumerated tiers with row padding, W8A16 + W8A8 combined-call variants, fail-loud errors, reproducible build.
