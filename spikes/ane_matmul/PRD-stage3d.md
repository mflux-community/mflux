# PRD — Stage 3d spike: row-split fused MLP on the ANE

**Branch:** `experiment/zit-on-ane-stage3d` (new, off `experiment/zit-on-ane-stage3`)
**Worktree:** `~/MFlux/mflux-ane-stage3` (or a fresh worktree for 3d — your call, say which)
**Report:** `spikes/ane_matmul/stage3d/3D-ROWSPLIT-MLP.md`
**Size:** small spike. Answer one question, then stop.

---

## 1. The question

> If the ANE takes a share of the **tokens (rows)** and runs the **whole MLP for those rows in one call**, does a real denoising step get faster?

## 2. Why we think it might

Stage 3c (`aba76b8d`) failed because of hand-offs, not compute:

- 4 ANE call sites per block × 34 blocks × 8 steps ≈ 1,100 GPU↔ANE waits per image.
- Each site paid ~6 ms of GPU-side overhead (activation quantise + transpose 2.9 ms, fp32 epilogue 3.0 ms).
- The 116 ms "block win" was cross-iteration overlap in a looped benchmark. In a real dependency chain, it vanished (153.2 vs 155.3 ms).

A row split changes the shape of the problem:

- **Tokens are independent everywhere except attention.** w1/w3 → SiLU gate → w2 is row-wise. The ANE can take its rows through the entire MLP without coming back to the GPU.
- **MLP hand-offs per block drop from 2 to 1** (w1w3 and w2 become one call).
- **The MLP is two-thirds of the block's linear FLOPs.** It's three 3840×10240 matmuls (~1.0 TFLOP at S=4224), against ~0.5 TFLOP for q/k/v/o.
- **W8A16 removes the activation-quantise step.** The ANE takes fp16 activations, plus int8 weights as inputs.
- **Recombining is a concat on the token axis.** There's no gather, and the share is tunable at runtime with no re-bake (only M changes).

The confidence level is reasoning, not measurement. That's why this is a spike.

## 3. Scope

### Phase A — fused row-split MLP, existing bridge (required)

One ANE program per padded row count `Mpad` (multiples of 256, enumerated):

```
in:  x_rows  fp16  [Mpad, 3840]   (already RMSNorm'd + modulated on GPU)
     w1, w3  int8  [10240, 3840]  (weightless: inputs, shared surface)
     w2      int8  [3840, 10240]
out: y_rows  fp16  [Mpad, 3840]   (MLP output, before gate/residual)
body: h = silu(x·w1ᵀ) * (x·w3ᵀ); y = h·w2ᵀ
```

- **Weights:** use the weightless / dynamic-weights path from `f16ba7bf`. Wired memory must stay flat.
- **Scales:** either fold them into a constant dequant inside the program, or apply per-channel scales in a GPU epilogue. Measure both if cheap. Otherwise use the 3c epilogue path, fused with the gate + residual on the GPU.
- **fp16 headroom:** `h` can overflow fp16. Reuse the ÷32 headroom trick before w2 and check the max abs value.
- **K = 10240 for w2:** inside the same program, chunk K into 4 and sum the chunks (as in 3b's K4). Above the ~2560–3072 cliff, the ANE needs K slicing.
- **Layout:** say which layout you used (`[1,C,1,S]` channels-first or matmul form). If a transpose is needed, fuse it with the modulation in one Metal kernel, or report its cost separately.
- **GPU side:** the GPU computes the remaining `S − Mpad` rows of the same MLP in bf16, in parallel. Concat on the token axis, then gate + residual.
- **Everything else stays on the GPU:** q/k/v/o, attention, norms and RoPE.

### Phase B — post-attention segment (optional, only if A passes its block gate)

Extend the ANE call to cover `o-proj → residual → norm → modulate → MLP → gate → residual` for its rows. That's one call per block after attention. Before attention, the q/k/v projections stay on the GPU for now.

### Out of scope

- A native async primitive (MLX custom primitive with `MTLSharedEvent`). That's the next spike if 3d shows promise.
- Attention on the ANE.
- W8A8.
- Packaging, CLI flags, upstream PRs.

## 4. Measurements

All timings use the **real dependency chain**: block N+1 consumes block N's output. No looping a single block or op. Use the same seed and the same prompt embedding as 3c.

| # | Test | Arms |
|---|---|---|
| T1 | MLP sub-block correctness | ANE share ∈ {0.15, 0.20, 0.25, 0.30} vs GPU bf16 |
| T2 | MLP sub-block time, chained over 34 blocks | GPU-only vs each share |
| T3 | Full block time, chained (Phase A wired in) | GPU-only vs best 2 shares |
| T4 | Step ladder: 8, 16, then 34 blocks + refiners | GPU-only vs best share |
| T5 | (Phase B only) T3 + T4 again | |

Record for every run:
- median and p90
- swap before and after (`sysctl vm.swapusage`)
- wired memory before and after
- peak RSS
- `macmon` ANE power (≈9 W means unthrottled)
- the split share and `Mpad`

Also break down the per-call overhead: transpose/prep, ANE wait, epilogue/concat.

## 5. Gates

| Gate | Pass | Kill |
|---|---|---|
| Correctness (T1) | cosine ≥ 0.9999 per block vs bf16; no inf/NaN | any inf/NaN that headroom can't fix |
| MLP in-chain (T2) | ≥ 15% faster than GPU-only MLP | < 5% |
| Block in-chain (T3) | ≥ 8% faster than GPU-only block | < 3% → stop, write up |
| Step (T4, 34 blocks) | ≥ 8% faster step; swap delta = 0; wired growth ≤ 1 GB | swap grows, or < 3% |

Between "kill" and "pass": report it and stop. No tuning loops past 2 iterations.

## 6. Benchmark discipline (unchanged from 3a–3c)

- **Reboot the Mac before the first timed run.** Stage 3c left ~16 GB of swap, and macOS won't release it without a restart.
- Run in the foreground or at Interactive QoS only. Never `ProcessType=Background`.
- Hold `s3a.LOCK` during timed runs. Keep `watchdog.sh` armed at 97% memory.
- **New quiet gate:** if swap grows during a measured run, the run is invalid. Discard it and report it.
- Interleave the arms (ABAB…), with at least 5 repeats per arm.
- Take every number and commit hash from tool output. Never type them from memory.

## 7. Permissions and constraints

- **Ask Scriv before:** any real-weight run, any image generation, any `git push`, or anything outside the worktree.
- Before any push, ask: "Do we run CodeRabbit locally first?"
- Push only to the fork. No upstream PRs from the spike.
- No GPL code from draw-things-community. ccv/s4nnc (BSD-3) may be adapted, but keep the copyright notice.

## 8. Deliverable

`stage3d/3D-ROWSPLIT-MLP.md`, containing:
1. A one-line verdict: GO / NO-GO / BORDERLINE.
2. Gate table with the numbers.
3. A per-call overhead breakdown, compared with 3c.
4. Swap and wired memory across all runs.
5. If NO-GO: which cost dominated, and whether a native async primitive would plausibly remove it (with numbers, not opinions).
6. Commit hashes.

## 9. Open questions for the agent to answer in the report

- Did the ANE compile the fused MLP as one ANE segment, or did `MLComputePlan` show a CPU/GPU fallback for SiLU or the multiply?
- Which share was best, and is it stable across the 8/16/34-block ladder?
- How much of the remaining overhead is sync (wait), and how much is GPU prep/epilogue?
