# Brief — Speeding up a diffusion transformer by running Apple's GPU and Neural Engine together

*A request for fresh ideas. We want approaches, not finished code. Short sketches are welcome: named APIs, ≤15 lines of pseudocode.*

---

## 1. The problem

**Goal:** make Z-Image Turbo image generation faster on Apple Silicon by using the **Neural Engine (ANE)** alongside the **GPU**. The speed-up has to work in a form that can ship inside an existing open-source app.

**Setting**
- **Hardware:** M4 Max, 36 GB unified memory, macOS 26. Targets: M3/M4. M5 is out of scope (its GPU has its own int8 units).
- **Software:** mflux, a Python app on Apple's **MLX** framework. The denoising loop is wrapped in `mx.compile` (MLX's graph compiler). The ANE is reachable only through **Core ML** (coremltools 9, MIL programs, `.mlpackage`). We can write native code as MLX **custom primitives** (C++/Objective-C, nanobind).
- **Model:** Z-Image Turbo's DiT (diffusion transformer, ~6B params).
  - Width 3840, 30 heads × 128, SwiGLU MLP 3840→10240→3840.
  - 34 transformer blocks: 30 main, plus 2 noise-refiner and 2 context-refiner blocks.
  - At 1024² there are **S ≈ 4,224 tokens** (image + text). Generation takes **8 steps**, with no CFG (the guidance is distilled in).
  - Text encoder: Qwen3-4B (one prefill pass). A VAE decoder at the end.
- **Where the time goes:** the DiT is the vast majority of image time. Within a block: linear layers ~80%, attention (SDPA) ~16%, everything else ~5%.

**The hard rules**
1. **Strict dependency chain:** step → block → layer. Block N+1 needs block N's output.
2. **Shippable** means it runs inside mflux's compiled MLX graph. We can't use free-running Python threads, because `mx.compile` forbids them.
3. **Memory:** it must fit in 36 GB with no swap growth during generation. Core ML models with baked-in weights *wire* memory (pin it, so it can't be paged out).
4. **Quality:** about as good as mflux's existing int8 (q8) path. Per-row int8 weights are acceptable.
5. **Licensing:** no GPL code. BSD/MIT is fine with attribution.

**Success** = a shippable path at **≥ 1.15× faster per denoising step** than the GPU alone (bf16), measured on a real chained step, not an isolated loop.

## 2. What we've tried, and what we learned

| # | Approach | Result | Why |
|---|---|---|---|
| 1 | GPU + ANE matmuls at the same time | Both kept ~full solo speed | Concurrency is real |
| 2 | Split one linear layer by output columns | 1.51–1.59× for that layer | Works in isolation |
| 3 | Fuse into 4 ANE calls per block, int8 weights baked in | Isolated block 158 → 117 ms | But ~1 GB wired per block, so the model won't fit |
| 4 | "Weightless" ANE programs (int8 weights passed as inputs, Draw Things style) | Wired memory flat (750 MB); ANE work 49 ms/block | Fixes memory |
| 5 | Weightless column split inside a **real chain** | 153 vs 155 ms per block, so **no gain** | The isolated 117 ms was overlap *between loop iterations*. In a chain, each hand-off stalls. ~6 ms GPU-side overhead per site (activation quantise/transpose 2.9 ms + fp32 epilogue 3.0 ms), × 4 sites × 34 blocks × 8 steps |
| 6 | **Row split:** the ANE takes a share of the tokens and runs the **whole MLP** for them in one call (W8A16, per-row int8) | Chained 34-block step **1.25×** — *threaded only, see #7* | Fewer, bigger hand-offs. Tokens are independent everywhere except attention |
| 7 | Same as #6, packaged as an MLX custom primitive (shippable) | **1.02×** | The primitive *blocks* during eval, so the GPU waits and there's no overlap. #6's 1.25× needed a Python worker thread that `mx.compile` won't allow |

**Other findings**
- **fp16 range:** the ANE accumulates in fp16 and silently *clamps* rather than producing inf. Real MLP activations reach ~5,000. We fixed this with a 1/√K constant in the dequant plus per-row activation scaling, restored in an fp32 GPU epilogue.
- **Memory duplication:** the GPU holds bf16 MLP weights and the ANE holds int8 copies, adding ~4.9 GB wired. The likely fix is one shared int8 copy that both units read.
- **QoS matters:** launchd's `ProcessType=Background` throttles the ANE (2.5–5 W vs ~9 W), making runs 1.5–1.6× slower.
- **Prior art:** Draw Things (whose ccv/libnnc library is BSD-3) reports ~1.4× from GPU+ANE on M3/M4, measured against its own int8 GPU path. It uses:
  - a per-GEMM row (token) split, with a cost model choosing the share;
  - weightless int8 programs per (Mpad, N, K), with M padded to 256;
  - weights blitted into shared IOSurfaces (CVPixelBuffer + `MLMultiArray initWithPixelBuffer:`), and outputs via `outputBackings`;
  - a fast fence plus a CPU spin-wait for sync.

  It's native Swift/C, not Python.

## 3. The open question

The overlap we need clearly exists: #6 shows 1.25×. But we can only capture it outside MLX's compiled graph. **How would you get that overlap in a shippable form, or get a similar gain by another route entirely?**

Our working assumption is a non-blocking primitive. **Please challenge it.** We especially want angles we haven't considered, such as:
- a different way to partition the work;
- a different place in the pipeline to use the ANE;
- a different integration point with MLX or Core ML;
- a different scheduling model;
- or a reason we're measuring the wrong thing.

## 4. What we'd like back

Your **top 3 ideas**, ranked. For each:
1. **Mechanism:** what changes and why it would capture the overlap or the gain. 2–4 sentences.
2. **Sketch:** named APIs or ≤15 lines of pseudocode. No full implementations.
3. **Cheapest test:** the smallest experiment that would prove or kill it, ideally under a day.
4. **Expected gain and main risk.**

Also, briefly:
- anything in this brief you think we've **got wrong**;
- any idea you'd **rule out** immediately, and why.

---

## Appendix — numbers (M4 Max 36 GB, 1024², from measured runs)

| Item | Value |
|---|---|
| GPU bf16 matmul throughput | ~12 TFLOPS. MLX q8 is ~14% slower on these shapes (in-kernel dequant) |
| Linear FLOPs per block (S = 4,224) | ~1.49 TFLOP (MLP ~1.0, q/k/v/o ~0.5) |
| Attention FLOPs per block | ~0.27 TFLOP |
| Block time, GPU bf16 | **~171 ms per main block** (5823 ms chained ÷ 34; 162 ms if the 2 refiners are counted). The earlier "~155 ms" came from Stage 3c's isolated block loop and is not a chained figure — 5.8 s ÷ 34 cannot be 155 ms |
| Chained 34-block step, GPU bf16 | ~5.8 s (measured 5814 and 5823 ms on the paired clean boot; 5844–6155 ms across boots) |
| ANE int8 work per block (dynamic weights) | ~49 ms (47.97 ms isolated at Mpad 2048; 46.98–54.72 ms as the `wait` term in-chain) |
| Weight copy into the shared surface, per call | ~0.28 ms **as a GPU blit**. A CPU `np.copyto` of the same 118 MB measures **5.1 ms**, so the figure is mechanism-specific and the two must not be mixed |
| Column-split overhead per site | quantise + transpose 2.9 ms; epilogue 3.0 ms |
| Row-split MLP, threaded (not shippable) | 5814 → 4634 ms, **1.255×** (paired, same boot) |
| Row-split MLP, blocking primitive (shippable) | 5823 → 5685 ms, **1.024×** (paired, same boot) |
| ANE power, unthrottled / throttled | ~9 W / 2.5–5 W |
| ANE power in the primitive path | bursty, median ~1.9 W when active (58 of 150 samples non-zero, max 2.50 W). Median over *all* samples reads 0.00 W and is misleading for a bursty signal |
| Accuracy, row-split MLP vs bf16 — **over the ANE's own rows** | min 0.99969, median 0.99991 |
| Accuracy, row-split MLP vs bf16 — over the full block | min 0.99985, median 0.99996. State the basis: the GPU rows are exact by construction, so the full-block figure dilutes the ANE error by roughly half |
| Wired memory: baked / weightless / row-split with duplicate weights | ~1 GB per block / 750 MB flat / +4.9 GB |
| Isolated vs chained block (column split) | 117 ms vs 153 ms. The isolated loop overlaps iterations |
