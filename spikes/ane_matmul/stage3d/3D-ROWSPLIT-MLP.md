# Stage 3d — row-split fused MLP on the ANE

**Verdict:** NOT RUN (code written on Linux, untested on a Mac). Fill in GO / NO-GO / BORDERLINE after the runs.

## What's here

| File | Purpose |
| --- | --- |
| `mlp_program.py` | MIL builder for the fused MLP (one program per Mpad), int8 quantiser, numpy fp16 mirror, split planner |
| `rowsplit.py` | `RowSplitFFN` / `RowSplitBlock` (ANE rows ‖ GPU rows, concat on tokens), backends, per-call timers |
| `bench.py` | T1–T4, ABAB interleave, swap quiet gate, wired/RSS/macmon capture, JSON results |
| `test_mlp_program.py` | numpy-only tests (mirror vs fp32, headroom, split plan, MIL builds). Passed on Linux. |

## Design notes (found while writing, verify on the Mac)

- **int8 inputs need the macOS 26 (iOS26) deployment target.** MIL `cast` rejects int8, and Core ML I/O turns int8 inputs into int32 below iOS26. Int8→fp16 is `dequantize` with a constant unit scale. `--weight-dtype fp16` is the fallback (2× the weight bytes of int8).
- **w1/w3 scales cannot go in a GPU epilogue.** SiLU is nonlinear, so they must be applied inside the program. Only the w2 scale can move to the epilogue (`--w2-scale epilogue`, needs `--scale-mode output`).
- **`--scale-mode weight`** dequantises rows in-program (safe fp16 range, extra elementwise ops on the ANE). **`output`** matmuls raw int8 values with x pre-shrunk by 1/128 and scales the output (cheaper, tighter range). Measure both.
- **Headroom:** the w3 branch is ×1/32 (same as `FeedForward.FLOAT16_HEADROOM`); the program returns y/32 and the epilogue multiplies by 32 in fp32.
- **K chunking:** w2 is 4×2560 summed in fp16 inside the program; w1/w3 default to one K=3840 matmul (`k13_chunks=2` if the ANE cliff bites).
- **Layout:** matmul form `[1, Mpad, 3840]`, weights `[out, in]`. No transpose needed.
- **Backends:** `coremltools` is correct but copies ~118 MB of weights per call, so it is only valid for T1. `bridge` is a stub: wire the `f16ba7bf` weightless bridge (its sources are not in this repo). `mirror` runs the numpy mirror to dry-run the harness.
- **`prep` timer** includes waiting on the upstream GPU chain (MLX is lazy), so it is an upper bound on true prep cost.
- Benchmarks use synthetic random weights at real dims: no downloads, no generation. Real-weight run: `patch_transformer` in `rowsplit.py`, needs Scriv's go-ahead.

## Run order

```bash
cd spikes/ane_matmul/stage3d
uv run python -m pytest -q test_mlp_program.py
uv run python bench.py t1 --backend mirror          # harness dry run
uv run python bench.py t1 --backend coremltools     # correctness + MLComputePlan report (open question 1)
# reboot, hold s3a.LOCK, arm watchdog.sh, then:
uv run python bench.py t2 --backend bridge --lock-file ../s3a.LOCK
uv run python bench.py t3 --backend bridge --shares <best two> --lock-file ../s3a.LOCK
uv run python bench.py t4 --backend bridge --shares <best> --lock-file ../s3a.LOCK
```

## 1. Gate table

| Gate | Pass | Kill | Result |
| --- | --- | --- | --- |
| T1 correctness | cos ≥ 0.9999, no inf/NaN | inf/NaN headroom can't fix | |
| T2 MLP in-chain | ≥ 15% faster | < 5% | |
| T3 block in-chain | ≥ 8% faster | < 3% | |
| T4 step (34 blocks) | ≥ 8%, swap Δ = 0, wired ≤ +1 GB | swap grows, or < 3% | |

## 2. Per-call overhead (ms, median) vs 3c

| | prep | ANE wait | epilogue/concat |
| --- | --- | --- | --- |
| 3c (per site) | 2.9 | — | 3.0 |
| 3d | | | |

## 3. Swap / wired / RSS / ANE power per run

| run | swap before→after (MB) | wired before→after (GB) | peak RSS (GB) | ANE W |
| --- | --- | --- | --- | --- |

## 4. Open questions

1. One ANE segment, or CPU/GPU fallback for SiLU/mul? (`compute_plan` in the T1 JSON)
2. Best share, stable across 8/16/34?
3. Remaining overhead: wait vs prep/epilogue?

## 5. If NO-GO: dominant cost and async-primitive estimate

## 6. Commits
