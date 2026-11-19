# Stage 3d — row-split fused MLP on the ANE

**Verdict: BORDERLINE — the timing premise is validated for the first time in this line of work,
but two gates fail.** A row-split MLP inside a real chained step is **1.265× faster** at 34 blocks,
and the win *grows* with ladder size (1.160× @8, 1.185× @16, 1.265× @34) — the opposite of Stage
3c, where the block win flattened to zero at step level. So the overlap 3c said needed a native
async primitive is reachable with a worker thread plus a GIL-released bridge call.
It is not a GO: **T1 accuracy fails** (cos 0.99980 vs the 0.9999 gate, in both scale modes, with the
intended math at 0.999967 — the shortfall is the ANE's own execution), and **T4 wired memory fails**
(+4.86 GB against a +1 GB allowance) for a structural reason a row split cannot design away.

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
   0.999967, so the gap is ANE execution, not int8. Cheapest discriminating experiment: T1 with
   `--weight-dtype fp16`, which removes quantisation entirely — one run, and it separates "the ANE
   mishandles this graph shape" from "quantisation interacts badly with the ANE". Cost if chosen:
   fp16 weights double the weight bytes to 236 MB/block (~8 GB host) and would likely raise the
   fixed ~8.9 ms/call, so it is a speed-for-accuracy trade, not a free fix. The 24% deviation in
   `weight` mode also deserves one focused probe with `a`, `b` and `h` returned as extra outputs —
   that pinpoints where the ANE diverges, and it is the single most informative unrun measurement.
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

## 6. Commits

- `f9231fc5` — spike code imported onto the 3d branch (cherry-pick of `ff8ecd9c` from
  `feat/dreamlite`)
- `4599aaa9` — fused MLP wired into the weightless bridge; `BridgeBackend` implemented; the
  `MLComputePlan` package-abort finding
- `f4474b6e` — memoise block quantisation (a share sweep was re-quantising all 34 blocks per share,
  ~16 GB of duplicate host int8 at four shares)
- `ce1cb694` — 128× gain fix in `output` mode; scale-sensitive assertions in test and T1; timed
  runs take the cooperative `BENCH.lock` via `s3a.BenchLock`
