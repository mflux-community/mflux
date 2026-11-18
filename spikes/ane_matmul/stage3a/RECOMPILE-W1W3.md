# w1w3 recompile check

Date: 2026-11-18, 20:42-20:51. Machine: M4 Max, 36 GB, macOS 27.0.1. Question from the mini PRD: why did the standalone w1w3 package run 95-103 ms in t4b? The same weights inside the multifunction package ran 53-55 ms.

Raw evidence: `results/recompile_w1w3.jsonl`. Script: `stage3a/recompile_w1w3.py` (commit `2a9086e0`). Run under `watchdog.sh` and the shared `BENCH.lock`. `STAGE3A-REPORT.md` was not edited.

## Answer

**The gap is gone, and it was never an artifact problem.** It was the bimodal machine state. In this check the slow state (~87-96 ms) landed on a different arm in each round. All four arms ran 52.8-55.1 ms in the fast state. The old compiled `.mlmodelc` itself runs at 53.1 ms when the machine is fast.

Per the PRD success criteria: the fresh standalone is within ~10% of the multifunction number (actual: within 2%). Treat the t4b 95-103 ms numbers as a state artifact. The 3b layer budget for w1w3 on the ANE stands at ~53 ms at T=4224 (20480x3840, W8A16, ~11.6 TFLOPS), for standalone and multifunction alike.

## Method

Four arms, interleaved, 6 rounds, best-of-3 (30 iterations) per arm per round:

| Arm | Artifact |
|---|---|
| A_mf | `mf_block.mlmodelc`, function `w1w3` (the t4 reference) |
| B_sa_oldc | `sa_w1w3.mlmodelc`, compiled during t4 (the "slow" artifact) |
| C_sa_fresh | package rebuilt from the same deterministic weights (seed 102), fresh compile |
| D_sa_recom | the old `sa_w1w3.mlpackage`, recompiled fresh |

Checks: weight-blob hash equality, artifact sizes and hashes, compute units, ANE state probes (3840x3840 c1 baseline) before and after.

## Results

Weights: the fresh package and the old package carry byte-identical weight blobs (sha256 prefix `a6fbe4f60c74f8e8`, `identical: true`).

Artifacts: all four arms plan to CPU_AND_NE (units "3" in every run). Sizes: mf 354.1 MB, each standalone 78.7 MB. Compiled `coremldata.bin` hashes differ between every pair, including D against B. Compilation is not byte-deterministic, and the hashes carry no performance meaning here.

Per-round medians (ms, best of 3 within each cell):

| Round | A_mf | B_sa_oldc | C_sa_fresh | D_sa_recom |
|---|---|---|---|---|
| 1 | **92.96** | 53.35 | **86.79** | 54.01 |
| 2 | **96.41** | **90.43** | 53.81 | 54.63 |
| 3 | 53.70 | 53.08 | 53.13 | 53.79 |
| 4 | 53.95 | 54.27 | 52.82 | 54.43 |
| 5 | 54.63 | 55.07 | 52.81 | 57.79 |
| 6 | 53.72 | 61.33 | 55.12 | 54.41 |
| **Best** | **53.70** | **53.08** | **52.81** | **53.79** |

Bold = slow state (~1.7-1.8x the fast state). Best-state spread across the four arms: 52.81-53.79 ms, within 1.9%.

State probes: the 3840x3840 baseline read 24.9 ms ("slow") before the A/B and 23.9 ms ("slow") after. The w1w3 arms still hit their fast numbers mid-run. The state flips per model run, not per time window. A single baseline probe does not classify the state of another shape's run.

## Readings

1. The t4b result (mf always fast, standalone always slow, 6/6 rounds each) was the machine state sticking to one arm for that session. It reproduced here in round 1-2 with the roles reversed: the multifunction arm was the slow one.
2. No stale-artifact effect exists. B (the exact artifact from t4b) runs at 53.1 ms best-state.
3. Multifunction packaging has no speed cost for w1w3: A best 53.70 against C best 52.81 (1.7%, inside the noise floor).
4. The machine was not fully quiet during this run (a co-tenant python bursted to ~99% CPU, Backblaze and desktop apps were active). The interleaving makes the comparison valid. The absolute fast-state numbers match the earlier quiet-window t4 measurement of 54.9-56.0 ms, so contention did not inflate them.

## Consequences for Stage 3b

- Keep the recommended layout: one multifunction package per block with enumerated tiers. This check removes the only measured argument against standalone packages, and neither layout has a speed advantage. Choose on artifact management grounds.
- Update the 3b layer budget: w1w3 ANE half ≈ 53 ms at T=4224 in the fast state (11.6 TFLOPS), not 95-103 ms.
- Every future benchmark on this machine needs interleaved arms and best-of-N. A single sequential best-of-3 block can still capture one arm in the slow state for its whole run, as t4b showed.
