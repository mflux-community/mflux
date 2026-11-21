# Stage 3a report: Z-Image ANE infrastructure tests

Machine: M4 Max, 36 GB, macOS 27.0.1. Worktree `/Users/ianscrivener/MFlux/mflux-ane-stage3`, branch `experiment/zit-on-ane-stage3`, base `experiment/zit-on-ane` at `3154afeb`. MLX 0.32.4, coremltools 9.0, Python 3.13.5. All weights are synthetic. All shapes are the production shapes. Token count is 4224 unless the test says otherwise.

Raw evidence: `spikes/ane_matmul/stage3a/results/*.jsonl`. Every number in this report comes from those files or from git output. Timing ran under `watchdog.sh` with the shared `BENCH.lock` (canonical path: the main checkout).

## Measurement method and machine state

During this session the machine showed a bimodal ANE state. The same 3840x3840 W8A16 model alternated between ~11-13 ms and ~20-26 ms medians, with p90 close to the median inside each run. The slow state persisted for minutes. It appeared under iCloud sync load, under a co-tenant agent, and once with no visible competing process. Stage 1 measured 9.1 ms for this shape at 4128 tokens on an idle machine.

Mitigation: every timed point is the best (lowest median) of 3 repeats. t2 keeps both halves of a split inside the same window, so a split pair always shares one state. t4b interleaves the two packages it compares. Absolute numbers may still sit above the idle-machine Stage 1 values. Relative comparisons inside one test are sound.

## Summary table

| # | Test | Result | Verdict |
|---|---|---|---|
| 1 | Fused QKV and w1w3 calls | Fused runs at equal or higher ANE efficiency. W8A8-pre fused: 14.1 TFLOPS, 1.09-1.14x against separate calls | Pass. Use fused, 4 ANE calls per block |
| 2 | Concurrency splits per shape | Best splits reach 1.31-2.04x against the GPU alone. Projected per-step gain ~1.55x | Pass |
| 3 | Enumerated token counts | Plans onto the ANE, numerically exact, same speed as fixed models at all 4 tiers | Pass. No per-tier files |
| 4 | Multifunction packages | Works, per-function load and speed at parity with standalones. No weight deduplication | Pass with a disk-cost caveat |
| 5 | Group-wise int8 matching MLX q8 | Representation near-lossless (4.7e-9, fp32 offset) but the conv plans to CPU, 24x slower | Fail for ANE use |
| 6 | Stream ordering | Ordering hack is best (1.27x vs GPU-only q8). No ordering: 0.80x. async_eval: 0.66x | Keep the hack, re-validate under mx.compile in 3b |
| 7 | W8A8 overflow safety | int32-like accumulation. The pre form (host quantise, post-matmul token scale) is overflow-safe. The in-graph dynamic form is not | Pass with a design rule |

**Go/no-go for Stage 3b: GO**, with the design rules at the end of this report.

## Test 1: fused shapes against separate calls

W8A16 (c1) and in-graph dynamic W8A8 (c1d), best of 3, T=4224:

| Case | Fused (ms) | Separate, one call (ms) | Separate, total (ms) | Fused TFLOPS | Separate TFLOPS |
|---|---|---|---|---|---|
| qkv W8A16 (11520x3840 vs 3x 3840x3840) | 34.5 | 24.9 | 74.8 | 10.8 | 5.0 |
| qkv W8A8-dyn | 67.9 | 29.6 | 88.7 | 5.5 | 4.2 |
| w1w3 W8A16 (20480x3840 vs 2x 10240x3840) | 57.9 | 35.3 | 70.6 | 11.5 | 9.4 |
| w1w3 W8A8-dyn | 116.6 | 49.5 | 99.1 | 5.7 | 6.7 |

Load note: the separate W8A16 qkv point never reached the fast state in its 3 repeats. It shows 5.0 TFLOPS against 10.8-12.6 elsewhere. The 2.17x total in the jsonl therefore overstates the fused gain. Read the TFLOPS columns.

Production W8A8 form (c1p: int8 input, token scale after the matmul), best of 3:

| Case | Fused (ms) | Separate total (ms) | Speedup | Fused TFLOPS | Separate TFLOPS |
|---|---|---|---|---|---|
| qkv W8A8-pre | 26.6 | 30.4 | 1.14x | 14.1 | 12.3 |
| w1w3 W8A8-pre | 48.1 | 52.2 | 1.09x | 13.8 | 12.7 |

Conclusions:

- Fusing never loses. The fused call runs at equal or higher TFLOPS and saves (n-1) Core ML dispatches per block. 4 ANE calls per block instead of 7 is achievable.
- The pre form is the fastest ANE path measured (13.8-14.1 TFLOPS against 10.8-12.6 for W8A16 in the same session).
- The in-graph dynamic W8A8 form is slow (5.5-6.7 TFLOPS). The quantisation ops run on the ANE as fp16 work. Do not use it (see also test 7).

## Test 2: concurrency splits per shape

Method: the GPU half (MLX) and the ANE half (Core ML) loop in a shared 8-second window after a 12-second ramp. layer_ms = max(gpu, ane) of the best window out of 3. GPU halves run in q8 (production default) and bf16.

Best split per shape and GPU mode:

| Shape (O x I) | GPU mode | GPU alone (ms) | ANE alone (ms) | Best GPU:ANE | Layer (ms) | Speedup |
|---|---|---|---|---|---|---|
| o 3840x3840 | q8 | 12.0 | 11.6 | 50:50 | 9.1 | 1.31x |
| o 3840x3840 | bf16 | 11.3 | 12.9 | 50:50 | 8.0 | 1.41x |
| qkv 11520x3840 | q8 | 35.3 | 33.3 | 50:50 | 17.9 | 1.98x |
| qkv 11520x3840 | bf16 | 31.8 | 32.1 | 60:40 | 19.3 | 1.65x |
| w1w3 20480x3840 | q8 | 65.1 | 77.9 | 50:50 | 31.8 | 2.04x |
| w1w3 20480x3840 | bf16 | 57.4 | 55.7 | 50:50 | 31.3 | 1.84x |
| w2 3840x10240 (k4) | q8 | 31.8 | 30.4 | 60:40 | 19.6 | 1.62x |
| w2 3840x10240 (k4) | bf16 | 29.0 | 50.4 | 60:40 | 17.7 | 1.63x |

Full per-split tables are in `results/t2_splits.jsonl` (48 points).

Observations:

- The GPU half does not slow down under ANE load. Example, w1w3 q8: GPU alone 65.1 ms at 20480 columns scales to 31.5 ms at 10240 columns inside the concurrent window.
- Bigger shapes gain more. The o shape gains least (1.3-1.4x) because its absolute times are small.
- w2 prefers a larger GPU share (60:40). Its ANE half carries the 4-chunk K-split overhead.
- q8 GPU solos are slower than bf16 solos at these shapes (65.1 vs 57.4 on w1w3). This matches the known mflux finding that q8 matmul trails bf16 at Z-Image shapes. The split layer times end up equal (31.8 vs 31.3), because the ANE half dominates the balance.
- Projection for 3b: assume the GPU half in q8 and the best splits. The four linear layers of one block then drop from 144.2 ms to 78.4 ms (1.84x). Linear layers are 79-80% of block time (Stage 1). That projects to ~1.56x per step against q8 alone. The PRD target is >=1.3x. Validate in 3d.
- One caveat: these splits come from free-running windows, not from the in-process joined layer. Test 6 shows the in-process path needs the ordering hack to reach overlap at all. Its measured speedup (1.27x on w2) is lower than the window projection (1.62x). Budget conservatively.

## Test 3: enumerated token counts

One function per layer shape with enumerated widths {4608, 6528, 8320, 16128}, against fixed-shape models per tier. w1 = 10240x3840, w2 = 3840x10240 with the 4-chunk K-split. Best of 3, iters 30:

| Layer | T | Fixed (ms) | Enumerated (ms) | Enum TFLOPS | Fixed TFLOPS |
|---|---|---|---|---|---|
| w1 | 4608 | 30.8 | 31.2 | 11.6 | 11.8 |
| w1 | 6528 | 75.5* | 43.0 | 11.9 | 6.8* |
| w1 | 8320 | 57.9 | 56.7 | 11.5 | 11.3 |
| w1 | 16128 | 107.1 | 106.2 | 11.9 | 11.8 |
| w2 | 4608 | 33.9 | 61.3* | 5.9* | 10.7 |
| w2 | 6528 | 91.3* | 46.2 | 11.1 | 5.6* |
| w2 | 8320 | 60.9 | 60.7 | 10.8 | 10.8 |
| w2 | 16128 | 118.0 | 122.2 | 10.4 | 10.7 |

`*` = that point's 3 repeats all landed in the slow machine state. Where both sides ran in the fast state, the enumerated model matches the fixed model within 1-4%. The enumerated w1 holds 11.5-11.9 TFLOPS at every tier.

- Placement: `aneplan` puts every op on the ANE for both enumerated models. The plans match the fixed plans. That includes all 4 K-split convs, slices and adds of w2.
- Numerics: cosine against fp64 is 0.9999999 at T=4608 and T=16128 (w1) and T=4608 (w2 k4). No inf.
- Load: 1.3 s (w1) and 3.2-3.6 s (w2) per process load.
- Caveat: a tiny probe model (64x128, tokens {256, 384}) planned its conv to CPU. Predict crashed (SIGTRAP) with ANE-enabled compute units, in Swift and Python. Real-size models show neither problem. Keep enumerated tiers at production sizes.

Conclusion: **enumerated shapes work. There is no need for per-tier files.**

## Test 4: multifunction packages

One `.mlpackage`, 8 functions at T=4224: qkv, o, w1w3, w2 in W8A16, plus W8A8-dyn twins built from the same weight arrays. Built with `mil.Program.add_function` + `export_as_multifunction`. Swift loads a function with `MLModelConfiguration.functionName`.

Sizes:

| Package | Size |
|---|---|
| mf_block (8 functions) | 354.1 MB |
| 4 W8A16 standalones | 177.0 MB |
| 8 standalones (a16 + a8) | 354.1 MB |

**No weight deduplication.** The multifunction package equals the sum of the standalones to within 7 KB. Each function stores its own weight copy, even when one numpy array fed both twins. Shipping W8A16 and W8A8 doubles the disk footprint in any layout.

Speed and load, interleaved A/B (t4b, 6 alternating rounds, W8A16, best state):

| Layer | mf best (ms) | standalone best (ms) | mf/standalone parity |
|---|---|---|---|
| qkv | 32.5 | 31.0 | 0.95x (mf 5% slower) |
| o | 12.4 | 12.8 | 1.04x |
| w2 | 29.5 | 29.7 | 1.01x |
| w1w3 | 52.9 | 95.5 | 1.81x — anomaly, see below |

Per-function load of the multifunction package takes 0.04-0.39 s. One 6.9 s outlier happened on the first touch after compile. Standalones load in 0.05-1.07 s, the same range. Loading one function does not pay for the others.

**w1w3 anomaly.** The standalone w1w3 package ran 95.5-103 ms in all 6 interleaved rounds. The same weights inside the multifunction package ran 52.9-55.0 ms in all 6 rounds. The same standalone measured 56.0 ms in the earlier t4 run. The state is deterministic per package within a session and unexplained. Candidate causes: a property of the compiled `.mlmodelc` artifact, or ANE residency effects when the runs alternate between two large models. Open item for 3b: recompile and re-measure before trusting any single package's absolute number.

Conclusion: multifunction packaging carries no measurable speed or load penalty and no dedup benefit. Choose it for artifact management (one file per block), not for size.

## Test 5: group-wise int8 against MLX q8

**Format discovery.** `mx.quantize(w, group_size=64, bits=8)` (affine, the mflux q8 format) stores:

- codes: uint8 in [0, 255], packed 4 per uint32 word along the inner axis.
- scale and bias: bf16, one pair per group of 64.
- Dequantisation is exactly `w = s * c + b`. Verified: after bf16 rounding the formula reproduces `mx.dequantize` with max error 0.0 on a 64x3840 probe.

**Core ML mapping.** `constexpr_blockwise_shift_scale` computes `scale * (data - offset)` per block. Set data = c (uint8), scale = s, offset = -b/s per group, block 64 along the inner dimension. Weight-level error against the MLX dequantisation, fp64, at 3840x3840:

| Offset dtype | Max absolute weight error | Relative to max abs weight (0.113) |
|---|---|---|
| fp16 | 3.7e-5 | 3.3e-4 |
| fp32 | 4.7e-9 | 4.1e-8 |

The representation is effectively lossless with fp32 offsets. Output numerics against an fp64 emulation of the MLX q8 matmul on identical fp16 input: cosine 0.99996, no inf or nan (`mx.quantized_matmul` itself: 0.999996).

**Placement and speed fail.** `aneplan` puts the blockwise conv on the **CPU**, not the ANE. The per-channel int8 conv with the same shape plans to the ANE. Best of 3 at 3840x3840, T=4224: blockwise 423.5 ms (0.29 TFLOPS) against per-channel 17.8 ms (7.0 TFLOPS), a 24x gap. Group-wise scales along the reduction axis force a weight decompression path that the ANE does not serve.

**Consequence for 3b.** ANE weights derived from a q8 checkpoint cannot keep the group-wise format. The PRD fallback applies: dequantise and requantise per-channel, with a logged warning. Stage 1/2 validated that per-channel path from bf16 at cosine 0.99999.

## Test 6: stream ordering

Mechanism, from MLX v0.32.4 source (`mlx/transforms.cpp` eval_impl, `mlx/scheduler.cpp`, `mlx/backend/cpu/encoder.h`, `mlx/backend/metal/eval.cpp`):

1. `eval_impl` builds a tape: DFS from a synchroniser node to count dependencies, then a width-limited BFS. Execution pops the tape from the back, so dependencies run before dependents, and siblings run in reverse push order.
2. CPU-stream nodes dispatch immediately. Each task goes on a per-stream FIFO worker thread as the walk reaches it. The ANE predict starts at that moment.
3. GPU-stream nodes encode into an open Metal command buffer. The buffer commits mid-walk only when a batching threshold trips (`encoder.needs_commit()`). Otherwise it commits at the end of `eval_impl` (`gpu::finalize`).
4. Cross-stream edges become fence waits recorded on the consumer stream. They do not block the host walk.

The concat join `[ya, yg]` pushes `ya` (ANE) before `yg` (GPU), and the reverse pop evaluates `yg` first. Whether the halves overlap then depends on when each device's work actually starts. Stage 2 measured serialisation inside the `mx.compile` predict path without an ordering dependency.

Eager-mode measurement, synthetic w2 (3840x10240), 4128 tokens padded to 4224, 50:50 split, GPU half q8, headroom 1/32, best of 3 alternating rounds:

| Strategy | Layer (ms) | vs GPU-only q8 (31.0 ms) |
|---|---|---|
| none (no ordering help) | 38.9 | 0.80x — serialised, worse than GPU alone |
| hack (GPU half consumes a zeroed ANE-input slice) | 24.3 | **1.27x** |
| `mx.async_eval(ya)` right after the ANE node | 46.8 | 0.66x — worst |

- All three strategies produce bitwise identical outputs (max diff 0.0).
- The hack stays the best mechanism. `async_eval` is counterproductive. It commits the ANE input chain as its own small GPU buffer and adds event synchronisation. The measured layer time ends up above full serialisation.
- Even in eager mode, the unhelped graph serialises (0.80x). The ordering dependency is not optional.
- 3b must re-validate under `mx.compile`, where Stage 2 saw the same serialisation. A cleaner long-term option: build the join so the ANE node is topologically first, with no fake dependency. For example, evaluate the padded fp16 input once, then construct the ANE node before any GPU consumer exists. The current hack costs one extra 1-element GPU op, which is negligible.

## Test 7: W8A8 overflow safety

**Graph structure (phase A, `aneplan` + MIL op dump at 3840x3840, T=64).** All three W8A8 forms keep every op on the ANE:

- `c1s` (static): `quantize(scale 0.02) -> dequantize -> conv -> identity`. The scale sits before the conv.
- `c1d` (dynamic per-token, in-graph): `abs -> reduce_max -> real_div -> maximum -> real_div -> round -> clip -> mul -> conv`. The graph computes the token scale and applies it before the conv.
- `c1p` (pre-quantised int8 input, production form): `dequantize(scale 1.0) -> conv -> mul(tok)`. The graph applies the token scale **after** the matmul. The build bakes the weight scales into the int8 weight constants at compile time.

**Accumulator width (phase B).** Adversarial probe: all weights +1 (int8, unit scale). Activations are +127 for the first half of the inner dimension, -127 for the second half. The exact answer is 0. The fp16 partial sums reach 243,840, far above the fp16 maximum of 65,504. Both the `c1p` int8-input form and the plain W8A16 form return **exactly 0.0**, with no inf and no nan. The ANE accumulates in a width that survives 24-bit partial sums (int32 or wider). It does not round through fp16 mid-reduction.

**Outlier tokens (phase C).** One token scaled 300x, realistic per-channel weight scales, T=64, 3840x3840:

- In-graph dynamic form (`c1d`): **12 inf** in the output. The dequantised activations enter the conv as fp16 values of order 1,500. The fp16 activation path overflows on the outlier token. Cosine against the fp64 emulation: NaN.
- Production form (`c1p`, host quantisation, int8 input, token scale after the matmul): **0 inf, 0 nan**. Cosine against the fp64 emulation: 0.99999991. Max |y| 1624 against a reference max of 1624.3.

**Design rule for 3b.** W8A8 must use the pre form: quantise activations on the host (GPU), send int8, apply the token scale after the matmul. Do not use the in-graph dynamic form. The int8 domain keeps partial sums below 2^31 for inner dimensions up to 10,240 (127 x 127 x 10240 = 165M). The post-matmul token scale keeps the fp16 output in range wherever the true output is in range. The 3b fp16 range audit (headroom 32 per the PRD) still applies to the scaled output.

## Recommended packaging layout

**One multifunction `.mlpackage` per DiT block, with enumerated token tiers {4608, 6528, 8320, 16128} in each function.** No per-tier files.

Evidence: enumerated models match fixed models on the ANE (test 3). Multifunction packages load and run at parity with standalones (test 4). Fused layer functions beat separate calls (test 1). Functions per block: qkv, o, w1w3, w2 — 4 ANE calls per block.

Conditions and notes:

1. The combination "multifunction AND enumerated shapes in one package" was not tested as a single artifact. Both halves work alone. 3b must validate the combination once, at build time.
2. No dedup: disk cost is the sum of the functions. W8A16 for one block is ~177 MB synthetic-scale (weights dominate: 44.3 + 14.8 + 78.7 + 39.3 MB). The full 32-block DiT projects to ~5.7 GB for W8A16, ~11.3 GB for both precisions. Shipping W8A8 doubles disk. Decide the default precision set with that in mind.
3. Per-block packages give cache granularity: one compiled `.mlmodelc` per block, rebuildable independently, matching the PRD cache-key design.
4. Fallback if the combination fails: per-tier multifunction packages (4 files per model), still no per-layer files.

## Go / no-go for Stage 3b

**GO**, with these design rules:

1. Default path W8A16, per-channel int8 weights, baked constants, channels-first 1x1 conv, fused qkv and w1w3, w2 with the 4-chunk K-split. 4 ANE calls per block.
2. Split ratios per shape (GPU:ANE): o 50:50, qkv 50:50, w1w3 50:50, w2 60:40. Hold them in config. The GPU half defaults to q8. bf16 (via `--dequantize-on-load`) measures slightly better on o and w2 layer times, and slightly worse on qkv.
3. Token handling: pad rows to the enumerated tier. One model per layer function covers all tiers.
4. q8 source checkpoints: dequantise and requantise to per-channel int8, with a logged warning. The group-wise format cannot run on the ANE (test 5).
5. W8A8: pre form only (host per-token quantisation, int8 input, post-matmul token scale). It is the fastest ANE path (13.8-14.1 TFLOPS) and overflow-safe (test 7). Image-level accuracy remains a 3d gate.
6. Stream ordering: keep the ordering-hack dependency. Do not use `mx.async_eval`. Re-validate in the `mx.compile` path early in 3b (test 6).
7. fp16 headroom factor 32 per the PRD (the Stage 2 hand-off said 16. The PRD wins).
8. Benchmarks in 3b/3d must use best-of-N and same-window pairing, and should note the bimodal machine state. Scriv's `powermetrics` run should cover a slow-state window to identify the cause.

Expected outcome if 3b integrates at the measured layer times: ~1.5-1.6x per step against q8 at 1024². That is above the PRD's 1.3x target. The main risk is the in-process join overhead. Test 6 measured 1.27x on w2 against the 1.62x window projection.

## Open items

1. The w1w3 standalone slowdown in t4b: 95-103 ms against 53-55 ms for the same weights in the multifunction package. The same standalone ran at 56 ms in the earlier t4 run. Deterministic within a session, unexplained. Recompile and re-measure in 3b.
2. The multifunction + enumerated combination artifact. Validate it once at 3b build time.
3. Root cause of the bimodal ANE state (thermal, ANE QoS, or a hidden co-tenant). Affects all absolute numbers on this machine.
4. Per-call Core ML dispatch overhead is still not isolated. t1 shows fusion of 3 calls into 1 saves ~13% at the qkv shape, which bounds overhead plus efficiency together.
5. The ordering hack under `mx.compile` (Stage 2 context) needs re-validation in 3b integration.
