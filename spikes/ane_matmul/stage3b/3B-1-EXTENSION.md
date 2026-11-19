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
inside the graph. The bridge transposes the caller's `(rows, I)` int8 buffer
into the channels-first array during the copy. It uses 64x64 tiles so both
tiles stay in L1. The W8A16 model keeps its token-major boundary with the
in-graph transpose. That path was already proven in test T1 below, and it keeps
the zero-copy input wrap, which int8 arrays do not support.

## MLX interface

The extension now binds two primitives over the same `AneModel`:

- `ane_linear(x fp16 [rows, I], model)` wraps the MLX buffer zero-copy.
- `ane_linear_q8(x int8 [rows, I], tok fp32 [rows], model)` copies and
  transposes into an owned int8 array and converts the scales to fp16.

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
| T4 timing, W8A16 enum tier 4608 (1920x3840) | 6.29 ms median |
| T4 timing, W8A8 pre enum tier 4608 | 9.25 ms median |

Every arm is finite and above the 0.9999 gate against the fp32 reference. The
W8A8 numbers compare against the same fp32 reference, so they include the
per-token quantisation error, and they still pass. The W8A8 call costs 2.96 ms
more than W8A16 at the same shape. That gap is the host transpose and copy of a
17.7 MB int8 buffer. A formal interleaved comparison belongs to 3b-2.

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
  `clip(round(x/tok))` to int8. T2 and T5 fix the numbers this must match.
- The extra 2.96 ms per W8A8 call at tier 4608 should be re-measured with the
  interleaved protocol before anyone cites it.
