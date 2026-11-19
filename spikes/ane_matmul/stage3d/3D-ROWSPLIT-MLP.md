# Stage 3d — row-split fused MLP on the ANE

**Verdict: NO-GO on real weights.** The gate Scriv set — ANE output mode ≥ mflux GPU q8, per block,
trained Z-Image-Turbo weights — fails in **0 of 90 rows**: ANE full-block cosine min **0.7753**,
median 0.9929, against q8 min 0.99992, median 0.99996 (§5e). Every number in §1–§5d was measured on
synthetic **init-magnitude** weights (std 0.016); the real FFN weights are std **0.17** with outputs
up to **2.6e6**, so §1–§5d stand as a graph and timing study, not as a shipping result.

The timing premise did survive testing, and that is a real finding: a row-split MLP inside a chained
step is **1.265× faster** at 34 blocks and the win *grows* with ladder size (1.160× @8, 1.185× @16,
1.265× @34) — the opposite of Stage 3c, where the block win flattened to zero at step level. So the
overlap 3c said needed a native async primitive is reachable with a worker thread plus a GIL-released
bridge call. It is moot for shipping, because the accuracy cost of running the MLP on the ANE is an
order of magnitude above the GPU q8 path already in mflux.

Machine: M4 Max 36 GB, macOS 27.2.1. Worktree `~/MFlux/mflux-ane-stage3`, branch
`experiment/zit-on-ane-stage3d` off `experiment/zit-on-ane-stage3` (`aba76b8d`); spike code
imported as `f9231fc5`. Synthetic seeded random weights at real Z-Image dims — no downloads, no
generation, no real weights. Every number below came from tool output.

## What changed since the spike was written

The spike said the `bridge` backend could not be wired because "the bridge sources are not in this
repo". That is wrong on this branch line: `f16ba7bf` carries `ane_bridge.h/.mm` and
`ane_linear.cpp`, so `BridgeBackend` is now real.

- `WLModel` gained a fused multi-input form (`set_fused` / `predict_fused`): seven graph inputs of
  arbitrary rank and dtype (fp16 `x_rows`, int8 `w1/w3/w2`, fp16 `s1/s3/s2`) and one fp16 output,
  all wrapped with `initWithDataPointer` / `outputBackings`. Additive only — 144 insertions, 0
  deletions — so the frozen 3a/3b/3c paths are byte-identical.
- `wl_fused_predict` **releases the GIL**, which is load-bearing. The caller runs the ANE on a
  worker thread while the main thread enqueues the GPU's rows; if the GIL were held, that enqueue
  could not start until the ANE finished and the harness would measure 3c's serial shape instead of
  the overlap 3d exists to test.
- `nanobind` had to be reinstalled (3.1.0) into `.venv`: the 3c build used `/tmp/ane_build`, which
  macOS had already cleaned.
- The 3d compile cache is `~/Library/Caches/mflux/ane_wl3d`, never `/tmp`.

## Corrections to the design notes (measured here)

- **`MLComputePlan.load_from_path` on a `.mlpackage` aborts the process.** It reads
  `<pkg>/coremldata.bin`, which weightless packages do not contain, and throws a
  `std::ios_base::failure` that no Python `try/except` can catch. It killed the first probe and
  cost a debugging cycle. It works on the compiled `.mlmodelc`, which is what the bridge already
  produces, so `compute_plan_report` now refuses a package by suffix.
- **`CoreMLToolsBackend` cannot run this program.** `MLModel.predict` raises `RuntimeError: value
  type not convertible` on int8 graph inputs, so the backend billed as "correct, valid for T1" is
  unavailable. Correctness here is the bridge against the real bf16 FFN, cross-checked against the
  numpy mirror.
- **int8 weight inputs at rank 2 are accepted** under the iOS26 target in matmul form with no
  transpose. That guess was right.
- **`output` mode carried a 128× gain error** (below). Timing was unaffected — the fix changed
  runtime scale values, not the op graph.

## The 128× bug, and why cosine hid it

`program_scales` folded `INPUT_SHRINK` (128) into `s2` as well as `s1/s3`. The pre-shrink applies
to `x`, which feeds `w1/w3`; `h` (the SwiGLU product feeding `w2`) is already restored by `s1/s3`,
so the extra factor made every `output`-mode program return 128× the true MLP output.
`epilogue_gain` carried the same factor for `w2_scale="epilogue"`.

| | gain_ratio | relerr | cosine |
| --- | --- | --- | --- |
| before | 0.0078 (= 1/128) | 126.87 | 0.99980 |
| after | 1.0008 | 0.0199 | 0.99980 |

Cosine is invariant to a constant gain, so it read 0.99980 either way. The blind spot was threefold:
the gate, the mirror (written from the same wrong reasoning, so it *agreed with the program about
being wrong*), and `test_mlp_program.py`, which asserted `cos > 0.999` and nothing about scale.
Fixed in all three: the fold, a relative-L2 assertion in the test, and T1 recording `relerr` and
`gain_ratio` and gating on them alongside cosine.

## Cost curves (one block, S=4224)

`ms/call` for one fused MLP over `Mpad` rows:

| Mpad | ANE `weight` | ANE `output` | GPU bf16, same rows |
| --- | --- | --- | --- |
| 256 | 23.36 | 13.67 | 5.63 |
| 512 | 31.44 | 18.52 | 10.57 |
| 768 | 41.11 | 22.70 | 15.69 |
| 1024 | 47.08 | 28.24 | 20.67 |
| 1280 | 55.65 | 32.77 | 25.87 |
| 2048 | 82.05 | 47.97 | 41.05 |
| 4096 | 146.52 | 85.37 | — |
| 4224 | — | — | 84.28 |

Fitted as fixed + marginal·rows:

| engine / mode | fixed (ms/call) | marginal (ms/row) | marginal TFLOPS |
| --- | --- | --- | --- |
| ANE `output` | ~8.9 | 0.0187 | 12.6 |
| ANE `weight` | ~15.2 | 0.0321 | 7.4 |
| GPU bf16 | ~0.6 | 0.0198 | 11.9 |

Two consequences. The ANE's *throughput per row* is essentially the GPU's — a row split does not
buy a faster engine, it buys a second engine, so the entire case rests on the two running
concurrently. And because the marginal rates are equal, small shares are bad: a small share adds
the fixed cost without removing much GPU work. That is why the PRD's 0.15–0.30 sweep looks flat and
the win appears at 0.50.

**Device attribution.** The compute plan reports only the *preferred* device, so throughput alone
could not separate the ANE from a CPU fallback. macmon over 25 calls at M=2048: ANE power median
**4.03 W** (max 4.62, idle 0.00), GPU 0.47 W, CPU 1.42 W. The ANE really is executing the program.
It is also well under the ~9 W the PRD treats as unthrottled, consistent with 12.6 TFLOPS rather
than the 13.8–14.1 TFLOPS 3b measured for W8A8-pre.

## 1. Gate table

| Gate | Pass | Kill | Result | Status |
| --- | --- | --- | --- | --- |
| T1 correctness | cos ≥ 0.9999, no inf/NaN | inf/NaN headroom can't fix | `output`: cos **0.99980**, relerr 0.0199, gain_ratio 1.0008, nonfinite 0 — identical across Mpad 512–2048. `weight`: cos **0.9708**, relerr 0.2398 | **FAIL** (both) |
| T2 MLP in-chain | ≥ 15% faster | < 5% | 1.313× @0.50, 1.186× @0.40, 1.004× @0.25 | **PASS** |
| T3 block in-chain | ≥ 8% faster | < 3% | 1.223× @0.50, 1.218× @0.40 | **PASS** |
| T4 step | ≥ 8%, swap Δ=0, wired ≤ +1 GB | swap grows, or < 3% | timing **1.265×** @34 (1.160× @8, 1.185× @16), swap Δ = 0; wired **+4.86 GB** | **PASS on time, FAIL on wired** |

T2 (34-block chained MLP, ABAB interleaved, 5 repeats, `output` mode, commit `ce1cb694`):

| arm | median ms | p90 ms | vs gpu_only |
| --- | --- | --- | --- |
| gpu_only | 3517.9 | 3619.1 | 1.000× |
| share 0.25 (m=1024) | 3504.5 | 3985.1 | 1.004× |
| share 0.40 (m=1792) | 2965.4 | 5969.2 | 1.186× |
| share 0.50 (m=2048) | 2678.9 | 3095.6 | **1.313×** |

Per block: 103.5 ms GPU-only against 78.8 ms split.

T3 (34 full blocks plus 2 refiners in the chain, same protocol):

| arm | median ms | p90 ms | vs gpu_only |
| --- | --- | --- | --- |
| gpu_only | 5894.8 | 6257.0 | 1.000× |
| share 0.40 (m=1792) | 4839.4 | 6067.1 | 1.218× |
| share 0.50 (m=2048) | 4820.3 | 5629.3 | **1.223×** |

Per block: 173.4 ms GPU-only against 141.8 ms split — 31.6 ms saved per block, 34 blocks. Note
173.4 ms/block here versus 3c's 155.3 ms base: different harness, so only arm-to-arm within a run
is comparable.

T4 (step ladder, share 0.50 only, m=2048, same protocol; each rung chains 2 refiners ahead of the
n main blocks):

| rung | gpu_only median ms | split median ms | speedup | split p90 ms |
| --- | --- | --- | --- | --- |
| 8 blocks | 1631.3 | 1406.1 | 1.160× | 1962.0 |
| 16 blocks | 2929.9 | 2472.0 | 1.185× | 2494.0 |
| 34 blocks | 5844.2 | 4620.2 | **1.265×** | 4750.0 |

The speedup rising with rung size is the headline: 3c measured the opposite (flat at 16 blocks,
1.8% *slower* than base). Per block at the top rung, 171.9 ms GPU-only against 135.9 ms split. The
8-block p90 (1962.0) is loose while 16 and 34 are tight, so the small rung is the noisy one, not the
large ones. Swap 0.25 → 0.25 MB with 0 runs discarded; wired 3.55 → 8.41 GB.

**Why the win now stacks when 3c's did not.** In a column split both engines must touch the same
activation, so each of the 4 sites per block is a serial stall inside the chain and the block cannot
finish before the ANE returns. In a row split the two halves are *data-independent*: rows `[0,m)`
belong to the ANE and rows `[m,S)` to the GPU, and they only rejoin at the token-axis concat. The
GPU therefore keeps working for the whole duration of the ANE call, which is what a real step needs.

**Why T1 fails, and why it is not quantisation.** The numpy mirror of the intended program agrees
with the real bf16 FFN at cos 0.999967 / relerr 0.0082 — that clears 0.9999. The ANE running the
same program lands at 0.99980 (`output`) and 0.9708 (`weight`). The residual is therefore the ANE's
own execution of the graph, and it is worst where the per-row scales are applied to the weights
*inside* the program (24% relative error, 30× the mirror's). Mechanism not established: separating
it needs intermediate outputs (`a`, `b`, `h`) dumped from the program, which would be a third
iteration and is past this spike's limit.

## 2. Per-call overhead (ms, median) vs 3c

| | prep | ANE wait | epilogue/concat |
| --- | --- | --- | --- |
| 3c (per site, 4 sites/block) | 2.9 | — | 3.0 |
| 3d T2 (MLP-only chain) | 0.49 | 55.30 | 18.17 |
| 3d T3 (full block chain) | 78.36 | 50.29 | 1.97 |
| 3d T4 rung 8 | 77.77 | 51.02 | 2.05 |
| 3d T4 rung 16 | 77.85 | 52.34 | 2.08 |
| 3d T4 rung 34 | 77.46 | 46.18 | 1.14 |

**Do not read these as three independent costs.** MLX evaluation is lazy, so the window a timer
covers depends on what the chain forces:

- In T2 the epilogue looks expensive (18.17 ms) because `mx.eval(out)` depends on `y_gpu` — the
  GPU's own work on the remaining rows — so the GPU tail is billed to the epilogue.
- T3 disproves that reading: with attention in the chain the epilogue is **1.97 ms**, at or below
  3c's 3.0 ms. The hand-off did get cheaper, as designed.
- In T3 `prep` is 78.36 ms because `np.array(h_ane)` forces evaluation of the upstream attention,
  which is genuine serial GPU work that must precede the FFN, not hand-off overhead.

Only totals are unambiguous, and the totals are the 1.313× and 1.223× above. MLP hand-offs per
block did drop from 4 sites to 1 as the design intended.

## 3. Swap / wired / RSS / ANE power per run

| run | swap before→after (MB) | wired before→after (GB) | peak RSS (GB) | ANE W |
| --- | --- | --- | --- | --- |
| probe, M=256 | 0.00 → 0.00 | — | — | — |
| t1 bridge, 5 tiers | 0.0 → 0.0 | 3.51 → 5.47 | 3.05 | — |
| t2 bridge, 34 blocks | 0.00 → 0.25 | **5.98 → 12.45** | 6.22 | — |
| t3 bridge, 34+2 blocks | 0.25 → 0.25 | **6.11 → 8.93** | 5.70 | — |
| t4 bridge, ladder 8/16/34 | 0.25 → 0.25 | **3.55 → 8.41** (+4.86) | 5.98 | — |
| device attribution burst | — | — | — | 4.03 med / 4.62 max |

No run was discarded by the quiet gate — swap growth stayed under the 1 MB threshold throughout,
because the machine was idle and exclusive.

**Wired memory is the design's main problem outside accuracy.** T2 grew wired by 6.47 GB and T3 by
2.82 GB, against a T4 allowance of +1 GB. Unlike the other findings this one is structural: the GPU
needs the full bf16 FFN weights to do the rows it keeps, so the int8 copies for the ANE are
*additional*, not a replacement — 34 blocks × 118 MB ≈ 4 GB of host int8, with kernel wiring on
top. 3c's column split could delete the offloaded columns from MLX, making a patched block cheaper
than a base block. A row split never can: both engines need every weight.

## 4. Open questions

1. **One ANE segment, or CPU/GPU fallback for SiLU/mul?** One segment. On the compiled model:
   65 ops, 30 on `MLNeuralEngineComputeDevice`, 35 const/none, **non-ANE ops: none** (`weight`
   mode); `output` mode 28 NE ops, also no fallbacks. SiLU and the SwiGLU multiply stay on the ANE.
2. **Best share, stable across the ladder?** 0.50 was best in T2 (1.313×) and T3 (1.223×), was the
   tightest p90 in both, and held across every T4 rung (1.160× @8, 1.185× @16, 1.265× @34). 0.40 was
   within noise of it in T3 but had the loosest p90 in T2 (5969.2 vs 3095.6). The optimum sits well
   above the PRD's 0.15–0.30 sweep because the marginal rates are equal — see the cost curves; at
   0.25 the split is neutral (1.004×), it only starts paying above ~0.4.
3. **How much of the remaining overhead is sync?** Little. At the T4 top rung the block costs 135.9
   ms, of which 46.18 ms is the ANE wait and 1.14 ms the epilogue. The GPU's own 2176 rows need
   ~43 ms, so the block is **ANE-bound**, not sync-bound: `wait` ≈ GPU time means the two engines
   really did run concurrently and the ANE call is now the critical path. The next lever is not the
   hand-off — it is making the ANE call itself faster (it held ~4 W, against ~9 W unthrottled, and
   3b measured 13.8–14.1 TFLOPS for W8A8-pre versus 12.6 here).

## 5. What continuing would take

Stage 3c recommended freezing the ANE lane, arguing that only a genuinely asynchronous primitive
could change the verdict. **That conclusion needs amending: a user-space async path already works.**
A worker thread, a zero-copy bridge call and a GIL release produced a chained win that grows with
step size. No `MTLSharedEvent` primitive was needed to get overlap. Two things block a GO, and they
are independent:

1. **Accuracy (T1).** cos 0.99980 vs a 0.9999 gate. The host mirror of the same program reaches
   0.999967, so the gap is ANE execution, not int8. Probes A, B and C (§5b, §5c, §5d) narrowed it
   considerably: each boundary `a`/`b`/`h` is accurate on its own; `k2_chunks` 1/2/4 are identical;
   quantisation is irrelevant, because fp16 weights read 0.97127 in weight mode (same as int8) and
   **0.95872 in output mode, which is worse than int8's 0.99980**; activation precision is irrelevant
   to seven decimals; and `output` mode holds 0.9998 for both weight sources while `weight` mode
   swings from 0.99953 (synthetic normal init) to 0.97073 (uniform `nn.Linear` init). `--weight-dtype
   fp16` is ruled out as a fix, and so is the magnitude sweep, because C already separated the
   variables it was designed to test. The sharpest remaining lead is **per-row scale uniformity**:
   the failing cell's weight rows all share max|w| = 0.01611, and the accurate cell spans
   0.04815–0.09396. Next experiment is therefore to perturb the row maxima of a uniform-init weight
   set and see whether `weight`-mode accuracy recovers. Until that is resolved, ship `output` mode
   (also the 1.7× faster one) and treat the residual as 0.99981 against a 0.9999 gate — still a fail.
   It must be re-measured on trained weights, which are neither uniform nor normal.
2. **Memory (T4 wired +4.86 GB).** Structural as written, but fixable by a design change: have the
   **GPU also consume the int8 codes** (dequantising on GPU) instead of keeping a bf16 copy, so one
   weight representation serves both engines. That removes ~236 MB/block of duplicate residency
   (~8 GB over 34 blocks) and, as a side effect, makes the two halves compute the same quantised
   weights — which would also tighten T1, since today the reference and the ANE disagree partly by
   construction.
3. **Scope honesty.** All of this is synthetic random weights at real dims. Real Z-Image weights
   carry outliers (3c measured a 300× case), so the accuracy margin should be expected to *shrink*,
   not improve, and any real-weight run needs Scriv's go-ahead and the memory watchdog — this
   harness alone costs ~6 GB of wired memory at 34 blocks.

A follow-up spike should answer 1 and 2 before any step-level integration work.

## 5b. Probe A: where inside the graph the ANE diverges (commit `f4010249`)

T1 fails, but §1 alone cannot say whether int8 quantisation or ANE execution causes it. Probe A
runs each graph boundary as its own single-output program through the bridge, and compares the
production output against both the bf16 MLX module (what T1 gates on) and exact fp32.

| boundary | cos vs exact fp32 |
| --- | --- |
| `a` = xW1ᵀ | 0.999963 |
| `b` = xW3ᵀ (returned as b/32) | 0.999962 |
| `h` = silu(a)·b/32 | 0.999889 |
| `y` = hW2ᵀ, k2_chunks = 1 / 2 / 4 | 0.999535 / 0.999535 / 0.999535 |

Four suspects are cleared, with `copybacks=0` throughout:

- **SiLU and the SwiGLU multiply are accurate.** The whole front half of the graph is fine on the ANE.
- **The 4-way K accumulation is not the cause.** 1, 2 and 4 chunks agree to the last digit.
- **T1's bf16 reference is not the cause.** bf16 module vs exact fp32 = cos 0.999990, relerr 0.0044.
- **Scale is not the cause.** Weight mode reads gain 0.9969 — the 128× bug is gone.

The failure is therefore in the ANE executing the graph, and it depends on where the per-row scale
is applied. Measured against exact fp32 with the corrected reference:

| mode | bridge vs fp32 | relerr | mirror vs fp32 | bf16 vs fp32 |
| --- | --- | --- | --- | --- |
| `weight` | **cos 0.970739**, gain 0.9969 | 0.2402 | cos 0.999977, 0.0069 | cos 0.999990, 0.0044 |
| `output` | **cos 0.999812**, gain 1.0008 | 0.0194 | cos 0.999977, 0.0069 | cos 0.999990, 0.0044 |

The host mirror of the same graph and the same weights reaches 0.999977. The ANE on the same
program lands at 0.9708 (`weight`) or 0.99981 (`output`). So the int8 representation is not the
blocker, and probe B (next section) removes it from the suspect list outright.

One result I could not reconcile, recorded rather than explained: probe A's own `y` arm reaches cos
0.99954 in weight mode, while the production graph reaches 0.9707 against the same kind of
reference. The op sequences look operationally identical, so the remaining variable is the data —
probe A uses synthetic normal weights and an fp16 activation, the production path uses bf16
initialised module weights and a bf16-quantised activation. Not established.

Probe A also found a Core ML constraint worth recording: a program must consume every input it
declares. Returning `a` while still declaring `w3` makes coremltools insert a placeholder
`identity` on the int8 weight, and compilation fails with "Param 'x' has incorrect type for
operator 'identity'". Any future boundary-splitting probe must declare only what it uses.

## 5c. Probe B: quantisation is not the cause either

B was meant to separate quantisation from graph execution by re-running T1 with
`--weight-dtype fp16`, where the weights enter as fp16 and no quantisation happens at all. Result
(5 shares, Mpad 512–2048, `gain_ratio` near 1 and `nonfinite` 0 in every row):

| weights | `weight` mode cos | `output` mode cos |
| --- | --- | --- |
| int8 | 0.97080 | **0.99980** |
| fp16, no quantisation | 0.97127 | 0.95872 |

Three conclusions, and the first one reverses what §5b claimed before this ran:

1. **Quantisation is exonerated.** Weight mode reads 0.9713 with exact fp16 weights against 0.9708
   with int8. The int8 codes are not producing the error.
2. **fp16 weights are not a fix — they are worse.** In `output` mode, int8 scores 0.99980 and fp16
   scores 0.95872, a 2.8% relative error. So "use fp16 to keep accuracy" is off the table, and it
   would also double the weight bytes.
3. **Small-magnitude weight operands are the bad case** — but probe C (§5d) showed magnitude is not
   the whole story:
   - int8 `output`: operand is raw codes, magnitude ~±127 → cos 0.99980 (best).
   - int8 `weight`: codes scaled by s_row inside the program, magnitude ~1e-2 → 0.97080.
   - fp16 `weight`: weights scaled by 1.0 (a no-op multiply), magnitude ~1e-2 → 0.97127.
   - fp16 `output`: same ~1e-2 operands → 0.95872.

   The three arms with small-magnitude weight operands all land at 0.959–0.971. The one arm with
   large-magnitude operands is an order of magnitude better. C then found two arms at the *same*
   magnitude differing by 2.9 points of cosine, so magnitude orders the failure but does not explain
   it on its own.

That is a **hypothesis from four measurements, not an established mechanism** — the two small-magnitude
arms differ from each other by more than a rounding argument explains, and probe A already showed
this probe's own `y` arm disagreeing with production. The confirming experiment is cheap and does
not need another dtype: scale the weight operand by a constant before the matmul and divide after,
sweeping the constant over 1/128, 1 and 128. If accuracy tracks magnitude rather than dtype, the
row-split MLP should ship the `int8` + `output`-scaled form (which is also the 1.7× faster mode) and
the accuracy gap left is 0.9998 against a 0.9999 gate.

Probe B also required a real fix: `BridgeBackend` hard-coded int8 for the weight inputs, so
`--weight-dtype fp16` would have mislabelled fp16 buffers to Core ML. It now derives the declared
type from `spec.weight_dtype` and stops forcing a dtype on the weight arrays.

## 5d. Probe C: the activation is irrelevant, the weight distribution is not

§5c attributed the failure to operand magnitude. Probe C tested the one variable that separated
probe A (0.99954) from production (0.9707) on a graph that is otherwise the same: the data. Cells =
{weights: synthetic normal | real `nn.Linear` bf16 init} x {activation: fp16 | bf16-rounded}, each
compared against exact fp32 of the tensors the ANE actually sees, so activation rounding is never
counted as ANE error. Mpad=1024, int8 weights.

| weights | act | `weight` cos | relerr | `output` cos | relerr |
| --- | --- | --- | --- | --- | --- |
| synthetic | fp16 | 0.9995341 | 0.0305 | 0.9998457 | 0.0176 |
| synthetic | bf16 | 0.9995340 | 0.0305 | 0.9998457 | 0.0176 |
| real-bf16 | fp16 | **0.9707341** | 0.2402 | 0.9998125 | 0.0194 |
| real-bf16 | bf16 | **0.9707305** | 0.2402 | 0.9998125 | 0.0194 |

Three conclusions, and the first kills two open questions at once:

1. **Activation precision is irrelevant.** fp16 against bf16-rounded differs in the seventh decimal
   (0.9995341 vs 0.9995340, and 0.9707341 vs 0.9707305). This also removes the discrepancy with
   probe A: probe A's activation was fp16 and production's was bf16, and that difference contributes
   nothing. The weight source is the remaining variable.
2. **The weight distribution drives the `weight`-mode failure.** Synthetic normal weights give
   0.99953; the real module's `nn.Linear` init gives 0.97073. The likely handle is in the last row
   of the probe's output: synthetic `w1` per-row maxima span 0.04815–0.09396, while the real ones
   span **0.01611–0.01611** — constant to five digits, because `nn.Linear` initialises uniform on
   `±1/sqrt(fan_in)`, so every row's maximum is essentially the same bound. Per-row scales that are
   nearly identical are the pathological input for weight-mode scaling on the ANE. Not proven, but
   it is the sharpest lead in the stage.
3. **`output` mode is robust and `weight` mode is not.** Output mode holds 0.99981–0.99985 across
   both weight sources. So §5c's magnitude story is incomplete: two arms with the same operand
   magnitude differ by 2.9 points of cosine, which magnitude alone does not explain.

**What this does not fix:** `output` mode still lands at 0.99981 against a 0.9999 gate, so **T1
remains FAIL**. The failure is now a small residual (relerr 1.9%) rather than a 24% structural one,
and the fast mode is also the accurate one, which is the useful outcome. Real trained weights are
neither uniform-init nor normal, so the residual must be re-measured before any real-weight run —
and if trained weights behave like the uniform-init cell in some layer, `weight` mode is unusable.

## 5e. Real-weight gate: ANE output mode vs mflux GPU q8 (the decisive measurement)

New gate, set by Scriv: **ANE output mode ≥ GPU q8**, per block, on trained weights. Weights come
from `Tongyi-MAI/Z-Image-Turbo` — its transformer ships as **F32** across three shards totalling
22.9 GB, so `real_weight_gate.py` reads three tensors per block through the safetensors index instead
of loading 23 GB. Reference is mflux's production path: the F32 checkpoint cast to bf16, run through
the real `FeedForward` (which shrinks the w3 branch by 32 and restores it in `to_stream`).

Arms per block, same input: the row-split ANE at share 0.50 (rows `[0,2048)` in the fused int8
program, `[2048,4224)` on GPU bf16) against `to_quantized(group_size=64, bits=8)`. Activation is
`ffn_norm1(N(0,1)) * g` for g ∈ {1,2,4}, standing in for `1 + scale_mlp`; real modulation needs the
text encoder and a denoising step, which this stage does not run, so g is a range, not a prediction.

30 blocks × 3 g = 90 rows, 92 s, no nonfinite anywhere:

| metric | ANE split | GPU q8 |
| --- | --- | --- |
| full-block cos | min **0.7753**, median 0.9929 | min 0.99992, median 0.99996 |
| owned-rows cos | min 0.7063, median 0.9865 | min 0.99992, median 0.99996 |
| ANE ≥ q8 | **0 of 90 rows**, on both bases | — |

Sample rows (block, g, ANE cos vs q8 cos): 0/1 0.9855 vs 0.99999; 0/2 0.8928 vs 0.99999; 0/4
**0.7921** vs 0.99999; 22/1 0.99989 vs 0.99992 (closest, still q8).

**Why the synthetic work mislead us.** Real FFN weights are std **0.17** with per-row maxima spread
w1 4.13–17.49 and w2 5.57–24.59, and real block outputs reach **2.6e6**. The synthetic probes used
init-magnitude weights (std 0.016) whose outputs peaked near 0.5. So every synthetic accuracy number
in §5b–§5d was measured in a regime roughly 10× smaller in weight magnitude and ~10^6–10^7× smaller
in output magnitude than production. In `output` mode the w2 matmul multiplies **unscaled h** by raw
±127 int8 codes, so its fp16 partial sums scale with the real activation magnitude and approach the
65504 fp16 ceiling — a failure mode the synthetic data could not reach. The error also grows with g
(block 0: 0.985 → 0.893 → 0.792), which is the signature of a range problem rather than a rounding
problem, and early blocks are far worse than late ones.

**Two of my earlier leads die here, and both died for the reason Scriv predicted.** The §5d
"near-identical per-row scales" lead was an artifact of `nn.Linear` uniform init: real weights are
the *most* spread of the three sets measured. And §5c's magnitude-of-weight-operand story is at best
partial once real magnitudes enter.

Not yet separated, and the one question worth another run: how much of the 0.9929 median is the
program's range design versus the ANE executing it. `probe_real_range.py` answers that by running the
numpy mirror over real weights — if the mirror reproduces the failure, it is graph design and a
headroom change could fix it; if the mirror stays accurate, it is ANE execution and no graph tweak
will.

## 5f. The mechanism: the program saturates fp16 on real weights, with no ANE involved

`probe_real_range.py` runs the **numpy mirror only** — no Core ML, no ANE — over real weights, and
prints the magnitude of each `w2` chunk partial sum against the fp16 ceiling of 65504. Blocks 0, 1, 5
at g=1 and every block at g=4:

| blk | g | max\|a\| | max\|h\| | max w2 chunk | chunk inf | mirror inf | mirror cos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 121.1 | 316.2 | **65504.0** | 425 | 209,330 | **nan** |
| 0 | 4 | 484.2 | 5060.0 | 65504.0 | 14,726,287 | 7,099,040 | nan |
| 1 | 1 | 66.4 | 123.4 | 65344.0 | 5 | 34,636 | nan |
| 5 | 1 | 61.1 | 138.6 | 65504.0 | 64 | 132,219 | nan |
| 15 | 1 | 78.5 | 87.4 | 25,840.0 | 0 | 0 | 0.9999144 |
| 29 | 1 | 69.9 | 84.2 | 18,976.0 | 0 | 0 | 0.9998002 |

**So this is graph range design, not an ANE quirk.** The mirror overflows in pure numpy: real `h`
reaches 300–5000, the `w2` matmul multiplies it by raw ±127 int8 codes, and a 2560-term partial sum
lands on the fp16 ceiling. `FLOAT16_HEADROOM = 32` is inherited from the bf16 GPU path, where it only
has to protect a bf16 accumulator; in fp16 it is one to two orders of magnitude too small. Blocks 15
and 29 at g=1 stay under the ceiling and are accurate, which is the control: the failure tracks
\|h\|, not block index.

Two consequences that matter beyond this stage:

1. **The ANE clamps instead of trapping.** The gate in §5e reported `nonfinite: 0` for every one of
   its 90 rows, because Core ML returns values pinned at the fp16 ceiling rather than inf or NaN. So
   the "no inf/NaN" condition in the original T1 gate is not evidence of range safety — a saturated
   program passes it. The mirror, which is honest arithmetic, returns `nan` for exactly the rows the
   ANE reported as clean-but-wrong. Any future gate must check output magnitude against the fp16
   ceiling, not only finiteness.
2. **The 128× bug from §5c was the same class of error**, and C's magnitude lead was groping at this.
   The real fix is the one §5c's `INPUT_SHRINK` was reaching for: shrink `h` before the `w2` matmul
   and fold that factor into `s2`, sized so `max|h| × 127 × sqrt(k_chunk)` stays well below 65504.
   At the observed `h` of 5060 that means a factor near 4096, not 32.

This does **not** overturn §5e's verdict. Even the clean blocks (15 and 29 at g=1) reach only 0.99991
and 0.99980 through the ANE against q8's 0.99993, so the ANE still loses the gate on every row. What
§5f establishes is *why* the worst rows are as bad as they are, and that a headroom change would
recover the saturated cases rather than needing an ANE rewrite.

## 5g. Item 2: the shared weight surface does NOT fix wired memory (my hypothesis was wrong)

Step 1 (`probe_wired_surfaces.py`) measured how the kernel charges for weight buffers, one program,
synthetic weights, Mpad=2048:

| arm | k | wired Δ | host int8 GB | ms/call |
| --- | --- | --- | --- | --- |
| distinct surfaces | 1 → 34 | **+0.57 GB** | 0.66 | 45.7 |
| one reused surface | 18 weight sets | +0.17 GB | 0.15 (plus 0.12 copy) | 47.1, of which **5.1 copy** |

So there is no per-model wiring multiple on this path — wired rises roughly **1:1 with the int8
bytes actually passed**. That explained T4's +4.86 GB arithmetically (34 blocks × 118 MB ≈ 4.0 GB)
and suggested a shared surface would fix it. I implemented `--weight-surfaces shared` (one surface
set, weights blitted in per call, serialized by a lock because each `RowSplitFFN` owns its own worker
thread), verified it produces **bit-identical** output to per-block, and re-ran T4.

Step 2 result — the fix does not work as implemented:

| T4 run | swap during run | wired Δ | 34-block speedup |
| --- | --- | --- | --- |
| per-block (`ce1cb694`) | 0.25 → 0.25 MB | +4.86 GB | 1.2649× |
| shared (`e323e8ac`) | 1160 → **4221 MB** | **+6.09 GB** | 1.1913× |

**Why it failed, stated plainly:** step 1 only measured the cost of *passing* 34 distinct buffers.
It did not test whether the 34 per-block int8 copies stop existing — and they do not.
`prepare_block_weights` still allocates 34 × 118 MB and `RowSplitFFN` still holds them, so the shared
surface added a 35th copy plus a 5.1 ms per-call memcpy and saved nothing. I got the cost without the
benefit. The probe was a valid measurement of a narrower question than the conclusion I drew from it.

The real fix is therefore the one §5 item 2 already names: **do not hold two representations**. Have
the GPU read the int8 codes (dequantising on GPU) so one set of weight bytes serves both engines.
That removes ~236 MB/block of bf16 residency rather than adding another copy.

**Timing caveat:** the shared run is contaminated. Swap grew ~3 GB during it, which per the
stage's own discipline invalidates timings, and one rung discarded a run. So 1.1913× is **not** a
shared-versus-per-block speed comparison; the machine had accumulated swap from earlier
real-weight runs and macOS does not release it without a reboot. The per-block 1.2649× from
§5e remains the clean timing number, and no further T4 timing work is meaningful until after a
reboot.

## 5h. Range and accuracy fix, searched in the mirror (no ANE), all 34 real blocks

`range_fix_mirror.py`, post-reboot (swap 0.00 MB). 34 FFN-bearing blocks = 30 `layers` + 2
`noise_refiner` + 2 `context_refiner`; the refiners had never been measured and they are the worst
blocks. 5 arms × 34 blocks × g ∈ {1,2,4} = 102 rows, 120 s. Gates: max |fp32 partial sum| **and**
max |stored fp16 value| ≤ 32752, and cosine over rows [0,2048) ≥ mflux q8 on the same block and
activation.

| arm | what changes | worst gated value | range ok | range + acc ok |
| --- | --- | --- | --- | --- |
| `cur` | today | inf | 15/102 | 0/102 |
| `k` | Draw Things constant 1/sqrt(K), real scales to the epilogue | inf | 85/102 | 0/102 |
| `hrow` | h divided by its own row max, restored in the fp32 epilogue | inf | 99/102 | 0/102 |
| `both` | k + hrow | **5,060** | **102/102** | 0/102 |
| `g64h` | group-64 weight scales + hrow | **5,084** | **102/102** | **101/102** |

Findings:

1. **a + b solves the range problem, and each alone does not.** `k` reaches inf on
   `noise_refiner.1` because its partial sums still scale with h; `hrow` reaches inf there because
   h, a and b are themselves stored in fp16 and exceed the ceiling before any normalisation.
   Together they sit at 5,060 against a 32,752 gate, and the value stops growing with g.
2. **The accuracy gate is separate, and per-row scales cannot pass it.** Every per-row arm tops out
   at cos 0.999984 against q8's 0.999993. That is not an ANE effect: it is granularity. One scale per
   output row is **60× coarser** than mflux q8's group_size=64, so the arm loses the gate however the
   range is handled.
3. **Matching q8 needs group-64 weight scales.** `g64h` clears range at 102/102 and beats q8 on
   101/102 rows. The single miss is `noise_refiner.1` at g=4: 0.9999997 vs 0.9999999 — a 2e-7 tie,
   not a defect.
4. `cur` peaks at 3.6e7 on `noise_refiner.1` (553× the ceiling), so the refiners belong in every
   future gate rather than being sampled from the main stack.

Two modelling bugs in this script's first version were mine: the rowmax restore was applied inside
the fp16 chain instead of the fp32 GPU epilogue (which turned refiner rows into inf and made a
range-safe arm look broken), and the gate covered only matmul partial sums, not the stored fp16
values of h and the program output. Both are fixed; the table above is the corrected run.

**Not yet shown on the ANE.** These are mirror numbers. The program must carry a group scale tensor
per weight ([out, K/64] rather than [out]) plus an h-rowmax input and its epilogue restore, and §5b
already proved the ANE can deviate from its own graph definition. So the confirming run is the next
step, not a formality.

## 5i. Step 1: the a+b range fix, built into the program and measured on the ANE

Implemented as `MlpSpec(range_fix=True)`, design following
`~/zCode_26_11/drawthings-ane-research/DRAWTHINGS-ANE-FINDINGS.md` (liuliu/ccv, BSD-3-Clause);
written from that description, no source copied.

- **(a)** each weight is widened with `mb.dequantize(scale = 1/sqrt(K_in))` instead of `1.0`, and
  the real per-row scale arrives on the matmul output as `s * sqrt(K)`. `x` is no longer shrunk.
- **(b)** `h` is divided by its own per-row max (`abs` → `reduce_max` → `maximum(6.1e-5)` →
  `real_div`), and that max leaves the program as a **second output** `h_max_out`, restored in the
  fp32 GPU epilogue. It cannot be restored inside the program: that is what overflows fp16.
- The bridge's fused call was generalised to N outputs (`set_fused(ins, outs)` /
  `predict_fused(in_ptrs, out_ptrs)`), rebuilt, and `run()` now returns `(y, h_max)`.
- Floor is `6.1e-5`, fp16's smallest normal: the natural `1e-8` underflows to zero in fp16 and would
  make a zero row `0/0`.

Smoke, Mpad=256 at real weight magnitudes: `h_max_out` shape (1,256,1), peak 11,448;
**peak |y_program| = 8.88** against a 65504 ceiling; `copybacks = 0`; compute plan 32 ops on the
NeuralEngine with **no non-ANE fallback**, so the row-max normalisation itself stays on the ANE.
Bridge vs mirror cos **0.9999995**. In the same script the *unfixed* arm returned **finite values with
cos 0.152 vs fp32** -- a third, independent confirmation that the ANE clamps rather than traps.

T1 over **all 34 real blocks** x g in {1,2,4} = 102 rows, share 0.50, Mpad=2048, 52 s:

| metric | value |
| --- | --- |
| range gate (peak \|y\| and \|h_max\| <= 32752) | **102/102**, worst 5,064 |
| copybacks / nonfinite | 0 / 0 |
| cos over ANE rows | min 0.9996898, median 0.9999133 |
| cos over the full block (GPU rows exact) | min 0.9998495, median 0.9999580 |
| **ANE minus mirror ('both' arm)** | min −9.5e-7, **median −7.4e-8**, max +4.6e-7 |
| rows within 1e-5 of the mirror | **102/102** |

**The fix does what it was meant to.** The ANE now reproduces its own graph on every block: the
largest ANE-vs-mirror gap is 9.5e-7, against the 2.7e-2 (weight mode) and 2e-4 (output mode)
deviations recorded in §5b and §5d. Those turn out to have been the saturation all along, not an
ANE quirk, and §5b's open question is closed.

The residual is **per-row int8 quantisation of the weights**, and it varies by block: worst is
`layers.27` at 0.9996898, which the mirror scores at 0.9996900 -- identical. So no further ANE work
will move that number; only finer scale granularity would, which is the `g64h` option Scriv dropped.

Per-block spread worth keeping in mind for Step 2: the refiners are the *most* accurate
(`noise_refiner.1` 0.9999996), the mid-stack main layers the least (`layers.24` 0.99977,
`layers.27` 0.99969).

**Memory bug found and fixed while doing this.** `prepare_block_weights` memoised by holding the
*source* arrays alive to make its `id()` keys safe, so 34 blocks pinned ~590 MB each (~20 GB) and
swap climbed to 6.3 GB. It now keeps a cheap content fingerprint (shape, dtype, first element) and
verifies it on lookup, with a 64-entry bound. Re-running T1: identical accuracy, 73 s to 52 s, and
swap flat (2970 to 2962 MB). Scope note: this leaked only where fresh numpy arrays are passed per
block. In `bench.py` the cache held references to MLX parameters the loaded blocks already own, so it
delayed release without duplicating memory -- the §3 and §5g wired numbers for T2/T3/T4 stand.

## 6. Commits

- `f9231fc5` — spike code imported onto the 3d branch (cherry-pick of `ff8ecd9c` from
  `feat/dreamlite`)
- `4599aaa9` — fused MLP wired into the weightless bridge; `BridgeBackend` implemented; the
  `MLComputePlan` package-abort finding
- `f4474b6e` — memoise block quantisation (a share sweep was re-quantising all 34 blocks per share,
  ~16 GB of duplicate host int8 at four shares)
- `ce1cb694` — 128× gain fix in `output` mode; scale-sensitive assertions in test and T1; timed
  runs take the cooperative `BENCH.lock` via `s3a.BenchLock`
- `f4010249` — probe A: graph-boundary programs and the two-reference gate (§5b)
- `16dc9cd0` — probe B: fp16 weights (§5c); `BridgeBackend` weight-dtype fix
- probe C (§5d) and this report update are committed together; `git log --oneline` gives the hash
