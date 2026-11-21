# 3B-1: the ane_linear extension, production-ready items

Date: 2026-11-19. Worktree `experiment/zit-on-ane-stage3`. This report covers the
items left after the assembly experiment (`3B-1-ASSEMBLY.md`): the combined-call
variants in W8A16 and W8A8, enumerated tiers with row padding, fail-loud errors,
and the reproducible build. All weights are synthetic. The run read no
checkpoint and generated no image.

## One Core ML finding drives the design

An enumerated int8 input must sit at a channels-first model boundary. A
token-major int8 boundary fails at prediction time with
`Invalid multi-array data type: 00020008`. The failure is in the Core ML plan
builder, not in our bridge. `diag_q8plan.py` proves it with five arms. The probe
in that script builds the `MLMultiArray` with the model's own input constraint,
the same call path the C++ bridge uses.

| Arm | Boundary | Shapes | Backings | Result |
|---|---|---|---|---|
| A | int8 (t, I), tok (t,) | fixed | no | predict OK |
| B | int8 (1, I, 1, t), tok (1, 1, 1, t) | fixed | no | predict OK |
| C | int8 (1, I, 1, t), tok (1, 1, 1, t) | enumerated | no | predict OK |
| D | int8 (t, I), tok (t,) | enumerated | no | plan fail |
| E | same model as D | enumerated | yes | plan fail |

Arms A to C pass numerics against hand math (`max_rel_vs_ref=0.000835`, fp16
rounding scale). D fails identically with and without output backings, so
backings are innocent. The fix: the W8A8 model takes `x int8 (1, I, 1, t)` and
`tok fp16 (1, 1, 1, t)`, and the final transpose to token-major output happens
inside the graph. The binding transposes the caller's `(rows, I)` int8 array
with `mx::contiguous(mx::transpose(x, {1, 0}))`, so MLX materialises the
channels-first `(I, rows)` buffer on its own device. The bridge then only does
one straight memcpy. The first cut did a blocked CPU transpose in the bridge.
That cost 2.96 ms per call and erased the W8A8 speed advantage, so the work
moved to the GPU. The W8A16 model keeps its token-major boundary with the
in-graph transpose. That path was already proven in test T1 below, and it keeps
the zero-copy input wrap, which int8 arrays do not support.

## MLX interface

The extension now binds two primitives over the same `AneModel`:

- `ane_linear(x fp16 [rows, I], model)` wraps the MLX buffer zero-copy.
- `ane_linear_q8(x int8 [rows, I], tok fp32 [rows], model)` transposes with an
  MLX op and memcpy-copies the materialised buffer. The caller keeps the simple
  row-major layout.

Both return fp16 `[rows, O]` written through `outputBackings`. `rows` must be
one of the enumerated tiers. The tier choice and the padding are the caller's
job, and `ane_models.pick_tier` implements it.

## Test results

`test_extension.py`, all synthetic weights, one run, emitted to
`results/extension_test.jsonl`. The run held `BENCH.lock`, but this
script is a correctness pass with a short timing loop, not a formal benchmark.

| Test | Result |
|---|---|
| Build both smoke models | 0.7 s |
| T1 W8A16 enumerated tier 4608, cosine vs fp32 | 0.9999628 |
| T2 W8A8 pre, max relative error vs fp64 emulation | 0.000986 |
| T2 cosine vs the W8A16 model | 0.9999626 |
| T3 fail loud, 5000 rows (not a tier) | RuntimeError raised |
| T3b 5000 rows padded to tier 6528, cosine vs fp32 | 0.9999627 |
| T5 fused QKV 11520x3840, W8A16 / W8A8 cosine | 0.9999625 / 0.9999253 |
| T5 o 3840x3840, W8A16 / W8A8 cosine | 0.9999624 / 0.9999253 |
| T5 fused w1w3 20480x3840, W8A16 / W8A8 cosine | 0.9999627 / 0.9999254 |
| T5 w2 3840x10240 K-split 4, W8A16 / W8A8 cosine | 0.9999577 / 0.9999159 |
| T4 timing, W8A16 enum tier 4608 (1920x3840) | 6.26 ms median |
| T4 timing, W8A8 pre enum tier 4608, GPU transpose | 5.63 ms median |

Every arm is finite and above the 0.9999 gate against the fp32 reference. The
W8A8 numbers compare against the same fp32 reference, so they include the
per-token quantisation error, and they still pass. The CPU-transpose build
measured 9.25 ms for the same call. Moving the transpose to MLX's device made
W8A8 faster than W8A16, and the formal comparison follows.

## Interleaved W8A16 against W8A8, production shapes

`measure_precision.py`, tier 4608, T = 4224 padded, 7 rounds, 15 calls per arm
per round, arms interleaved within each round, timed from the shared bf16 input
through the layer output (each arm pays its own cast or quantise work). Raw:
`results/precision_ab.jsonl`, `results/precision_macmon.jsonl`. BENCH.lock held,
watchdog used. Paired per-round ratios (W8A8 / W8A16, median of 7):

| Layer | W8A16 ms | W8A8 ms | Ratio | Ratio spread |
|---|---|---|---|---|
| fused qkv 11520x3840 | 30.81 | 26.43 | 0.8603 | 0.0467 |
| o 3840x3840 | 12.15 | 11.63 | 0.9444 | 0.0509 |
| fused w1w3 20480x3840 | 52.23 | 43.36 | 0.8302 | 0.0105 |
| w2 3840x10240 ksplit 4 | 32.08 | 27.87 | 0.8729 | 0.0394 |

W8A8 wins at every shape, by 6 to 17 percent. The absolute values put the ANE
at full rate: w1w3 W8A16 at tier 4608 gives 13.9 TFLOPS, and the Stage 3a
unthrottled fixed-4224 figure was 14.4. The macmon mean reads 6.41 W because
each round alternates ANE calls with GPU quantise work, so the ANE rail is not
busy for every window. The peak sample is 7.98 W. The benchmark paired every
ratio inside one round, so machine state cannot explain the ranking. Note for
the product plan: the
`q8 is slower than bf16` finding applies to the GPU. On the ANE the W8A8 conv
path is genuinely faster than W8A16.

## Fail loud

`predict` and `predict_q8` record the Core ML error text in `last_error()` and
throw. The MLX scheduler rethrows the stored error at the next `mx.eval`, so a
silent fallback is impossible. Example from the run:

```
RuntimeError: predict failed: MultiArray Shape (5000 x 3840) was not in
enumerated set of allowed shapes
```

The binding also rejects wrong dtypes or shapes before dispatch
(`ane_linear_q8: x must be int8 [rows, in_features]`).

## Reproducible build

`ops/build_extension.sh` is the one-command build. It pins nanobind 3.1.0 into
`/tmp/ane_build/nanobind` with `uv pip install --target`, resolves `MLX_DIR`
from the venv's installed MLX share directory, configures CMake against
`.venv/bin/python`, builds, copies the `.so` into `ane_linear/`, and ends with
an import check. It passed on the current tree. nanobind 3.1.x matches the MLX
0.32.4 ABI (v22). A rebuild after each C++ edit is one command, so the two
failed-rebuild stop rule from the PRD never triggered.

## What 3b-2 must carry forward

- Pad to `pick_tier(rows)` and slice the output back. The padding test shows the
  tier-6528 path works. Tiers 8320 and 16128 build the same way. The tests
  exercised only two tiers so far.
- W8A8 needs the host quantise per call: row `amax/127` in fp32, then
  `clip(round(x/tok))` to int8. The transpose is the binding's job now. T2 and
  T5 fix the numbers this must match.
- `--ane-precision w8a8` is the faster ANE mode at every production shape.
  Keep W8A16 available for the accuracy margin (cosine 0.99996 against 0.99992
  to fp32) and for calls that skip quantise overhead at small row counts. The
  `o` layer is the closest call at ratio 0.94.
- Assembly priority items (a) to (d) are in `3B-1-ASSEMBLY.md`. V0 (column
  concat) won at 118.8 ms per block, within 4 ms of the 115 ms target, and the
  W8A8 gain above should close the rest inside the split block.
