# 3B-2 dynamic weights: the Draw Things form works here, and it passes the gate

Date: 2026-11-19. Worktree `experiment/zit-on-ane-stage3`. Synthetic weights,
rebooted machine, MEMGATE on, macmon and in-run canary, one writer.
Design follows `~/zCode_26_11/drawthings-ane-research/DRAWTHINGS-ANE-FINDINGS.md`,
which describes liuliu/ccv (BSD-3-Clause, copyright Li Liu). No ccv source is
copied. `bench/surface_run.swift` carries the attribution notice.

## Correction to the earlier note

The previous conclusion "dynamic int8 weights are unreachable through
coremltools" was wrong on one point. The converter does accept the Draw Things
program form: `mb.dequantize` with a **constant** scale keeps the input int8,
and the dequantised fp16 operands feed `mb.matmul`. The rejected forms are
int8 straight into matmul or conv, and dequantize with a **dynamic** scale.
That leaves exactly the ccv shape: weightless programs take int8 in, apply the
constant `1/sqrt(K)` inside, and keep the real scales outside in a GPU epilogue.

## Part 1: weightless programs, speed and numerics (owned arrays)

One program per call site, padded M=4352 (4128 rounded up to 256, not the
4608 tier), K sliced at the ccv cliff (2560 to 3072). Use 2 slices at K=3840 and 4 at
K=10240). Convert times are small: 0.1 to 0.7 s per shape.

| Site (N,K) | weightless ms | cold first call ms | baked W8A16 ms | baked W8A8 ms | weightless vs W8A16 | cosine vs exact |
|---|---|---|---|---|---|---|
| qkv (5760,3840) | 11.80 | 16.3 | 17.92 | 11.74 | 0.66 | 0.9999999 |
| o (1920,3840) | 4.85 | 5.6 | 6.05 | 6.61 | 0.80 | 0.9999999 |
| w13 (10240,3840) | 19.90 | 26.5 | 25.16 | 20.00 | 0.79 | 0.9999999 |
| w2 (1920,10240) | 12.48 | 13.6 | 32.29 | 18.21 | 0.39 | 0.9999999 |

ANE work per block: weightless 49.0 ms against baked W8A16 81.4 ms and baked
W8A8 56.6 ms. The weightless form matches baked W8A8 per call and beats baked
W8A16 clearly. 13.2 to 17.2 TFLOPS, macmon 2.0 to 3.6 W. The cosine is against
an exact float64 emulation of the int codes path. The 0.48 cosines in the
round-1 notes were a scale-orientation bug in that probe, fixed here.

## Part 2: IOSurface handoff and wired memory

`bench/surface_run.swift`: one process holds all four programs plus one
activation, weight, and output IOSurface per shape
(`OneComponent8` / `OneComponent16Half`, packed rows,
`MLMultiArray(pixelBuffer:shape:)`, `outputBackings` on the output surface).
The process holds 34 synthetic layer weight blobs in ordinary memory. Each call
copies the layer weights into the shared weight surface, then predicts.

The known-answer check proves the transport: codes w=1, x=2 give y=2.002
against expected 2.0, so no stride bug hides in the pixel-buffer views.

| Cycle (34 blocks x 4 sites x 5 iters = 680 calls) | value |
|---|---|
| predict ms median / p90 | 8.0 / 14.1 |
| weight copy ms median / p90 (memcpy) | 0.28 / 0.91 |
| wired delta above idle baseline, median / max | 750 MB / 1063 MB |
| macmon ANE peak | 3.9 W |
| Metal blit variant, wired median | 3401 MB, rejected |

The blit variant wires every source buffer (`makeBuffer(bytesNoCopy:)` over
the heap blobs becomes resident). Use memcpy. It is faster here too.

## Gate: PASS

Wired: about 1.0 GB max against the 3 GB budget, flat with layer count. The
per-model weight bake was the memory wall, and the weightless form removes it.
Per-call: every site at or below baked, most far below the minus-20% bar.
Compile: 0.1 to 0.7 s per shape, 5.6 to 26.5 ms first call. Copy: under 1 ms.

## Stage 3c plan (weightless bridge inside the column split)

1. Bridge: a Swift/C++ primitive owning four cached programs keyed on
   (Mpad, N, K), one shared surface set per key, weight blobs referenced per
   layer. Eval: memcpy layer codes, predict into the output backing.
2. GPU side (MLX): keep the column split and the padded shared input rules
   from 3b-2. Quantise tokens to int8 and transpose into the activation
   surface layout. ccv fuses these, and a plain transpose plus quantise pair is
   the starting point. Epilogue multiplies the raw y by the token scale and the
   per-channel weight scale, transposes back, and joins the GPU-half columns.
3. Weights: at load, quantise each ANE-side column block to int8 codes plus
   fp16 channel scales once, keep them pageable, drop the columns from MLX as
   now. Estimated footprint: about 3 GB ordinary memory for all 34 blocks,
   about 1 GB wired, no 128-model plan set.
4. Padding: 256-aligned M per resolution instead of tiers. A second Mpad costs
   under 1 s of first-call compile, so 512x512 (Mpad 1280) becomes viable.
5. Validation order: block cosine against baked W8A8 (expect at least 0.9999),
   block time against 124 ms, then the step ladder to 34 blocks under the
   MEMGATE. Only then revisit the real-weight stages 3b-3 and up.
6. Open risks for 3c: fp16 epilogue rounding with real outlier activations
   (Stage 3a overflow tests must be rerun on this form), the token transpose
   cost at 4352 rows, and whether keeping W8A16 (baked) as an option for
   small-batch quality still matters.

## Verdict

Not a freeze. The memory wall that killed 3b-2 is a property of baked
weights, and the Draw Things form removes it at no speed cost. Propose Stage
3c: the weightless bridge described above.
