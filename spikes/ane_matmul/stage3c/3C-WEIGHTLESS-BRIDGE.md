# Stage 3c: the weightless bridge is correct, but it does not save time. Freeze recommended.

Date: 2026-11-19. Worktree `experiment/zit-on-ane-stage3`. Synthetic weights
only. We did not download or convert any real weights. Machine: M4 Max 36 GB.
All timed runs used MEMGATE, macmon, an in-run bf16 matmul canary, and
interleaved arms. The canary sat at 55.2 to 55.7 ms in every accepted round,
which is the clean band.

## Verdict against the three gates

| Gate | Target | Result | Status |
|---|---|---|---|
| Block cosine vs baked W8A8 | ≥ 0.9999 | worst 0.9999947 | PASS |
| Block time | ≤ ~115 ms | best 153.2 ms (share 0.30) | FAIL |
| Step ladder, 34 blocks | no swap growth, step win | flat at 16 blocks, 16 GB swap at 32+refiner | FAIL |

Two gates fail. The failure is structural, not a tuning miss. Details follow.

## What this stage builds

1. `wl_programs.py` — one weightless program per call-site shape (Mpad, N, K)
   in the Draw Things form: int8 weight `[1,1,N,K]` plus int8 token
   `[1,1,K,Mpad]` inputs, a constant `1/sqrt(K)` dequantize inside, fp16
   matmul, K sliced 2 or 4 above the 2688 cliff.
2. `ane_bridge.h/.mm` — `WLModel::predict` wraps all three buffers with
   `initWithDataPointer` (int8 wrapping works) and `outputBackings`. Zero
   copies in steady state. A copy-back on any input raises and the binding
   fails loud.
3. `wl_layers.py` — split linear, fused split, attention and feed-forward
   wrappers for the block, with the real per-token and per-channel scales in
   an fp32 MLX epilogue.
4. `~/Library/Caches/mflux/ane_wl` — compile cache keyed by
   `wl_{Mpad}_{N}_{K}_os{maj}_ct{ver}`. It stores the `.mlpackage` and the
   compiled `.mlmodelc`. Warm load is 1.12 s per program. Convert is 0.1 to
   0.7 s per shape. Not `/tmp` (macOS deletes `/tmp` contents).
5. `wl_step_battery.py` — the ladder now drops the ANE-owned columns from
   MLX after patching, so a patched block costs less memory than a base
   block, not more.

## Gate 1 detail: correctness (PASS)

Block 15 of a synthetic transformer at T=4128, Mpad=4352. Five GPU:ANE
shares, repeated iterations. Every weightless arm sits at 0.999994 or better
against baked W8A8 and against base. The outlier retest passed too: 300x
outlier tokens through the weightless path give cosine 0.9999757 and finite
output. The largest raw dot divided by K measured 92.4, far under the fp16
limit of 65504. The quantizer bounds each code to ±127, and the program
dequantizes with a constant scale.

## Gate 2 detail: block time (FAIL)

Same block, 6 rounds x 8 iterations, interleaved, memory gate clean at start
(79% free, 1.1 GB swap):

| Arm | median block ms |
|---|---|
| base (MLX only) | 155.3 |
| baked W8A8 control | 126.3 |
| baked W8A16 control | 128.8 |
| weightless share 0.50 | 167.1 |
| weightless share 0.45 | 165.7 |
| weightless share 0.40 | 161.2 |
| weightless share 0.35 | 158.1 |
| weightless share 0.30 | 153.2 |

The weightless block gets closer to base as the ANE share shrinks. At share
0.30 it matches base within noise. It never approaches 115 ms. The probe
result from 3B-2 still holds: each weightless ANE call is as fast as baked
W8A8 per call. The block result shows what the probe did not measure: the
cost around each call.

Per site, the weightless path adds about 2.9 ms of GPU quantise plus
transpose, a full drain before the blocking ANE call, and about 3.0 ms of
fp32 epilogue. The measured reference: quantise 2.726 ms, quantise with
transpose 2.911 ms, epilogue 2.974 ms. Four sites add roughly 24 ms of extra
GPU work per block. The blocking call also stalls the GPU: the ANE runs
alone, then the GPU runs alone, so the two never share the block time. The
baked path keeps no per-call quantise work, and its enumerated tiers amortize
the input handling, so it still wins at block level.

## Gate 3 detail: step ladder (FAIL)

Compiled whole-step runs of the synthetic transformer at latent (16,1,128,128),
T=4128, canary clean in every row:

| Row | step median ms | peak RSS | wired after patch |
|---|---|---|---|
| base, 0 blocks patched | 4996 | 0.4 GB | n/a |
| weightless, 8 blocks | 5069 | 10.2 GB | freed 9.8 GB during patch, drift -0.04 GB over 8 passes |
| weightless, 16 blocks | 5088 | 17.1 GB | same pattern |
| baked W8A8, 16 blocks (from 3b-2) | 4949 to 5007 | 3.7 to 6.8 GB | n/a |
| weightless, 32 blocks + refiner | not run | — | killed: swap reached 16.3 GB |

Two facts matter. First, the memory change worked: patching frees the original
columns (wired delta during patch: -9.8 GB), and wired stays flat across
eight timed passes. The weightless path no longer has the baked path's
1 GB-per-block wall. Second, the step does not get faster. The 16-block
weightless step is 1.8% slower than base, and the 16-block baked step from
3b-2 was also flat (4949 to 5007 against 4989). The block-level win does not
stack into a step, for either form.

## Why the block win does not stack

A step is a chain of data-dependent blocks. Inside the chain, each ANE call
must finish before the next operation can read it, and the next block cannot
start before this one ends. The standalone block harness overlaps consecutive
loop iterations, so iteration n+1 runs GPU work while iteration n waits on
the ANE. That overlap exists only in the loop. A real step has none of it.
The consequence: at step level, offload costs ANE time plus sync overhead and
buys only idle GPU time back. Both the baked ladder and the weightless ladder
show the same flat result, so this is not a defect of the bridge. It is the
shape of the problem.

## Items from the GO list, and where they landed

1. Re-sweep the split ratio: done at block level (0.50 down to 0.30). The
   optimum sits at the edge, toward the GPU. More ANE share is worse, which
   matches the serial-cost explanation above.
2. Double-buffer the weight copy: moot. The bridge wraps MLX buffers in
   place, so no weight copy exists in steady state.
3. Compile cache on disk: done, warm load 1.12 s per program.
4. Outlier retest on the weightless form: done, safe (see Gate 1).
5. Transpose plus quantise fusion cost: measured, 2.9 ms per site. It is
   real but small. The drain and the serial stall around the blocking call
   dominate.
6. Keep a W8A16 quality option: the baked W8A16 path from 3b-2 remains
   untouched in `stage3b/` and in the `--ane` integration.
7. BENCH.lock path mismatch: resolved. Every timed runner imports
   `BenchLock` from `s3a.py`, so one canonical path exists:
   `spikes/BENCH.lock` under `mflux-upstream`.

## Bug found on the way

`q8_layers.FeedForwardSplit` scaled the fused h3 input by 1/32 but restored
only the ANE columns, so the GPU columns of w2 came out 32x small. Corrected
and committed this session. It slightly understated the baked-vs-base cosines
in the 3b-2 tables (about 4e-4). Arm-to-arm comparisons there stand.

## Recommendation

Freeze the ANE line at Stage 3c. Do not proceed to real weights (stages
3b-3 and beyond). The bridge is correct, cheap on wired memory, and fast per
call, and it still cannot beat the GPU, because the wins only exist when the
GPU and the ANE work at the same time, and a step of a diffusion transformer
does not give them that chance without extra machinery.

The only path that could change the verdict is a genuinely asynchronous ANE
primitive: the ANE call runs on a worker stream, the GPU continues with all
work that does not read the ANE output, and the dependency joins later. That
is extension redesign work, not tuning, and it would need its own go/no-go
gate before any real-weight stage.
