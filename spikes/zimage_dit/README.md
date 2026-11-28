# Z-Image DiT cost harness

Synthetic benchmark scripts from the Z-Image DiT non-matmul cost investigation. Both ran on an M4 Max
with 36 GB, on 2026-11-17 and 2026-11-18. They answer two questions. How much of a DiT block is the
linear layers? What does the all-True attention mask that `transformer.py` used to build cost?

Nothing here loads real weights. Weights are random. `spikes/` is scratch. The package does not import
it, and no test imports these files.

Result JSON files are not committed. `.gitignore:16` has a repo-wide `*.json` rule with an explicit
allowlist, and these outputs do not belong on it. Every number they held is written below.

## Read this first: the block-level mask saving did not reproduce

| Run | MLX | reps | block with mask | block without mask | mask cost |
|---|---|---|---|---|---|
| A | 0.32.0 | 15 | 161.09 ms | 159.16 ms | +1.21% |
| B | 0.32.4 | 25 | 159.17 ms | 161.80 ms | **-1.63%** |

Both at S=4224, bf16. Run B is faster *with* the mask than without it. Two runs disagree in sign, so
the block-level effect is inside this method's run-to-run band, which is about plus or minus 1.5% of
block time. The honest statement is that the mask's block-level cost is not distinguishable from zero
here. The PR originally claimed 1.21% of block time and about 1% of a step. Neither is reproducible,
and both are corrected in the commit message and the release note.

What does reproduce is the isolated attention call. `sdpa_mask_bench.py` measured each cell 40 times,
across three sequence lengths and both MLX versions, and the masked path was slower in every cell. So
the mask is a real, small, consistent cost inside SDPA. Attributing a step-level percentage to it needs
a whole-model measurement on real weights, which was never run here.

## `sdpa_mask_bench.py`

Times one `mx.fast.scaled_dot_product_attention` call at Z-Image's real attention geometry: batch 1,
30 heads, head_dim 128, bfloat16. It compares mask forms at three joint sequence lengths. Every masked
form is bitwise identical to `mask=None`. That result is exact rather than statistical: an all-True
mask becomes an all-zero additive one, and adding 0.0 to a bfloat16 score is exact.

Median of 40 reps, on `mx.compile`d closures, MLX 0.32.0. Percentage is against `mask=None`:

| S | `mask=None` | additive zeros | bool broadcast (1,1,1,S) | full (B,H,S,S) bool |
|---|---|---|---|---|
| 1152 | 2.108 ms | +2.3% | +2.9% | +9.4% |
| 4224 | 26.044 ms | +2.6% | +2.7% | +11.0% |
| 5952 | 51.604 ms | +3.0% | +3.5% | +12.8% |

MLX 0.32.4 gives the same picture, at +2.5% to +4.3% for the broadcast forms.

A third run, in this branch's own venv and stamped with its commit, reproduces the
direction and widens the band slightly downward:

Stamp: `mlx 0.32.0`, `python 3.13.5`, commit `ed376061`, branch
`investigation/z-image-dit-nonmatmul`, 40 reps, percentage against `mask=None`.

| S | `mask=None` | additive zeros | mask built in graph | bool broadcast |
|---|---|---|---|---|
| 1152 | 2.128 ms | +2.0% | +3.7% | +3.8% |
| 4224 | 26.870 ms | +3.5% | +3.0% | +2.3% |
| 5952 | 51.182 ms | +2.9% | +2.9% | +2.8% |

Across all three runs, the additive mask that production built cost 2.0% to 4.3% of the attention
call. The masked path was slower in every cell of every run. That reproducibility is why the PR
claims this number, and not the block-level one.

Three things worth knowing from this table. A materialised per-head mask costs four times the broadcast
one, so any future mask must stay `(1,1,1,S)`. The kernel reaches about 10.5 TFLOP/s at 4224 and 5952,
and 9.6 at 1152, which is matmul-class throughput. So attention is not inefficient here.
`mask="causal"` is a useful control, but only once its work is counted correctly. A causal mask keeps
the lower triangle, so it contributes `S(S+1)/2` score entries rather than `S*S`. An earlier version of
this script scored the causal row with the dense count. It reported about 20 TFLOP/s. That was an
artefact of the arithmetic, not a faster kernel. With the triangular count, from a 15-rep run on
MLX 0.32.4:

| S | causal ms | causal TFLOP/s | dense ms | dense TFLOP/s |
|---|---|---|---|---|
| 1152 | 1.243 | 8.21 | 2.124 | 9.60 |
| 4224 | 13.675 | 10.02 | 25.824 | 10.61 |
| 5952 | 27.240 | 9.99 | 51.089 | 10.65 |

At the two long sequences causal takes roughly 0.53 of the time. Scored on its own work it runs
within about 6% of the dense rate, which is the half-the-work ratio the model predicts. At 1152 it sits about 14% below, where
the fixed cost per launch is no longer amortised. So the control holds at the lengths that matter here,
and it is weaker at the short one.

## `block_linear_share.py`

Times one real `ZImageTransformerBlock`, imported from mflux rather than copied, at `dim=3840`,
30 heads, head_dim 128 and hidden 10240. It reports the whole block, the block reduced to its seven
matmuls plus the SwiGLU activation, and the SDPA call alone. Activations are passed as traced
arguments, so `mx.compile` cannot fold the measurement away. Weights stay captured, as in production.

The stable output is the linear share. All rows are MLX 0.32.4, bf16, median of 15 reps except the
1024x1024 row, which is 25:

| Resolution | S | Whole block | Linear only | SDPA | Everything else |
|---|---|---|---|---|---|
| 512x512 | 1152 | 41.93 ms | 38.18 ms, **91.1%**, 10.9 TFLOP/s | 2.17 ms, 5.2% | 1.58 ms, 3.8% |
| 1024x1024 | 4224 | 159.17 ms | 129.54 ms, **81.4%**, 11.8 TFLOP/s | 27.45 ms, 17.2% | 2.18 ms, 1.4% |
| 1024x1456 (~1.5MP) | 5952 | 248.05 ms | 178.62 ms, **72.0%**, 12.1 TFLOP/s | 52.67 ms, 21.2% | 16.76 ms, 6.8% |

Run A put the 1024x1024 linear share at 78.7% on MLX 0.32.0, so the versions agree to about 3 points.
The trend across resolutions is the real finding. Linear work dominates at 512x512. Attention grows
with the square of the sequence length. By ~1.5MP it takes a fifth of the block, and the linear share
falls to 72%.

The last column answers the question the investigation was opened with, and it is the least reliable
number here. It is a leftover: the block minus the linear term minus the attention term, so it absorbs
the error of both. Treated as one quantity it spans 1.4% to 6.8% across runs of configurations that
should not differ, which is a factor of nearly five. Read it as an order of magnitude and nothing
finer. The defensible statement is an order of magnitude. Work that is neither a linear layer nor
attention is a few percent of block time. It is not the half that motivated the investigation. An earlier draft of this
work quoted 4.7% as a measurement. That precision was not there to be had.

**q8 is slower than bf16 at 1024x1024.** Run B gives the block at 172.56 ms against 159.17 ms. The
linear path is 142.06 ms against 129.54 ms, so +10% and +9.7%. Run A gave +11.6% and +13.9%. Both runs
agree on the sign, which is the part that matters. Size any offload or quantisation claim against the
bf16 numbers, because the baseline that would ship is the faster one.

Weights are 362 MB per block in bf16 and 192 MB in q8, so a step reads 12.30 GB or 6.54 GB across 34
blocks. Apple publishes 410 GB/s for this 36 GB M4 Max, and that figure was not measured here. On it
the read costs 30 ms or 16 ms, which is why the step is compute-bound rather than bandwidth-bound at
this size.

One cross-check agrees with this harness. The agent working on the Apple Neural Engine offload reported linear layers at
3.76 s per step across 30 blocks. That figure came from their own harness, on real weights. This one
gives 129.54 ms x 30 = 3.89 s, or 3.81 s on run A.

## Commands

Run from the repo root. `uv run` picks the pinned project environment. That environment has moved
during this work. `pyproject.toml` on `main` pinned `mlx==0.32.0`, and `experiment/zit-on-ane` now
pins `mlx==0.32.4` at the maintainer's request. Which branch an environment was synced from therefore
changes the numbers. Every table below names its MLX version. Only the stamped third SDPA run
carries a commit too. Runs A and B predate that rule. They were taken from the shared checkout's venv,
which belongs to whichever branch was synced there. They are labelled accordingly, rather than given a
provenance they do not have.

```sh
uv run python spikes/zimage_dit/sdpa_mask_bench.py --reps 40
uv run python spikes/zimage_dit/sdpa_mask_bench.py --reps 40 --eager   # also time the uncompiled path
uv run python spikes/zimage_dit/block_linear_share.py --reps 25 --bits 16,8 --grids 1024x1024
```

`--grids` is a case-sensitive substring of `512x512`, `1024x1024` or `1024x1456 (~1.5MP)`. Run one grid
per invocation. Graph compilation, not timing, dominates the wall clock.

Repeat before believing a block-level percentage. A single run of `block_linear_share.py` does not
resolve differences below roughly 2% of block time. That is exactly how the 1.21% claim failed.

## Taking the lock

Take `spikes/BENCH.lock` in the shared checkout before any timed run. Other agents benchmark on this
same Mac at the same time. Two runs contending for GPU and memory bandwidth corrupt both sets of
numbers, and a corrupted number looks exactly like a real one.
