# Server retracing in the Z-Image `mx.compile` graph

## The problem

`ZImage._predict` returns `mx.compile(predict)` (`variants/z_image.py:263-265`). The DiT sequence
length is image tokens plus caption tokens (`transformer.py:137` on `main`, `:139` after PR #1001).
Image tokens depend only on resolution. Caption tokens depend on prompt length, and `_patchify`
pads them up to a multiple of 32 (`transformer.py:201` on `main`, `:202` after PR #1001). A new
caption length therefore produces a new input shape.

One hardware exception matters for any measurement of this. `variants/z_image.py:263-265` returns the
uncompiled `predict` when `AppleSiliconUtil.is_m1_or_m2()` is true. That predicate tests for "max" or
"ultra" first. It returns false when either appears (`utils/apple_silicon.py:15-17`), so it is true
only for the base and Pro members of those two generations. Four chips never reach `mx.compile`: M1,
M1 Pro, M2 and M2 Pro. Every other chip takes the compiled path, including M1 Max, M1 Ultra, M2 Max
and M2 Ultra. A first-step measurement on one of those four says nothing about `mx.compile`
retracing, because that build never traces at all.

MLX recompiles a compiled function when an input shape changes. The `shapeless` option exists
specifically to stop that: `mx.compile(fun, shapeless=True)` is documented in
`mlx/core/__init__.pyi:4789` as "will not be recompiled when the input shape changes". A server that
serves varied prompts can pay a fresh trace on most requests without it.

No measurement of the retrace cost exists yet. Measure that before choosing an approach.

## Option 1: compile shapeless

Blockers are the places where a shape becomes a Python int. This branch is based on `main` and does
not carry PR #1001, so the primary numbers below are `main`'s. The column on the right is where each
read moves once PR #1001 lands.

Reads that change with caption length, so these are the ones that matter per request:

| read | `main` | with PR #1001 |
|---|---|---|
| `cap_ori_len = cap_feats.shape[0]` | `transformer.py:200` | `:201` |
| `batch_size, seq_len, _ = hidden_states.shape` | `attention.py:53` | unchanged |
| `batch_size, seq_len, n_heads, head_dim = x.shape` | `attention.py:99` | unchanged |

Reads that change only with resolution, so they are stable across prompts on a fixed-size server:

| read | `main` | with PR #1001 |
|---|---|---|
| `x_len = x_emb.shape[1]` | `transformer.py:136` | `:138` |
| `C, F, H, W = image.shape` | `:207` | `:208` |
| `image_ori_len = image.shape[0]` | `:216` | `:217` |

Three further reads exist on `main` today and disappear with PR #1001, because they existed only to
size the all-True masks: `x_emb.shape[0]` at `:108`, `cap_emb.shape[0]` at `:124` and
`unified.shape[1]` at `:139`. Two of those three varied with caption length. So on `main` the traced
path has nine such reads, and six survive PR #1001. Count the three mask reads until that PR lands.

Start with the three caption-dependent reads. The two in `attention.py` only build reshape targets,
which is the easiest case. Whether MLX accepts them under `shapeless` tracing is untested, so try it
first and read the error before refactoring anything.

That is the whole benefit of PR #1001 for this problem: it removes a third of the blockers, and the
two caption-dependent ones among them. It is not a fix on its own, because `cap_feats.shape[0]` and
the two `attention.py` unpacks still vary with prompt length.

## Option 2: bucket the caption length

Bucketing the padding modulus inside `_patchify` does **not** work. The trace key is the shape that
enters the compiled function, and `_patchify` runs inside it (`transformer.py:97`, padding at
`:200-201`). Coarsening the modulus therefore buys nothing on its own. Take 33 tokens and 45 tokens.
`(-33) % 32` is 31 and `(-45) % 32` is 19, so both pad to 64 and share a bucket. They still arrive at
`predict` as `(33, D)` and `(45, D)`, which are two trace keys. The padding has to happen before the
compiled call, on `text_encodings` in the caller, for bucketing to reduce traces at all.

That is a wider change than it sounds, because `cap_ori_len = cap_feats.shape[0]` (`:200` on `main`,
`:201` after PR #1001) is also how
the model learns which caption rows are real. If the caller pads first, those pad rows become real
rows: `cap_pos_ids` spreads over them, and `cap_pad_mask` marks none of them. So the caller must pass
the true length alongside the padded tensor, or the padding and position work has to move out of
`_patchify` altogether. Either way this is a signature change, not a constant change.

Two further costs. Extra pad tokens run through the whole network, and attention cost grows with the
square of the sequence length. Also, pad tokens are not excluded from attention. They take the pad
token embedding and then attend and are attended to, so a larger bucket changes the output image, not
just the runtime. A real pad-excluding mask would be needed to make bucketing output-neutral, and that
is a model-behaviour change rather than a caching change.

If such a mask is ever added, keep it broadcast-shaped. Measured at S=4224, an additive or boolean
mask of shape `(1,1,1,S)` costs about 2.6% of the attention call, while a materialised
`(B,H,S,S)` mask costs about 11%.

## How to measure it

Compare the first denoising step against later steps for a caption length the process has not seen.
That isolates trace cost from step cost. A prefill timer already records `prefill_s` and
`step1_excl_prefill_s` behind `MFLUX_TIMING=1`, but it is on the `improvement/z-image-perf-uplift`
branch (`src/mflux/models/z_image/variants/z_image.py`), not on `main`. Base this work on that branch,
or land its timer first. The driver script those fields fed was `tools/z_image_prefill_bench.py`.
That file is an untracked local tool. It has no commit anywhere in this repo, so it is not in the
tree.

Also confirm whether a new `mx.compile` wrapper per `generate_image` call reuses the previous trace.
`_predict` builds a fresh compiled function on every call (`variants/z_image.py:115`), so the
per-request cost may be larger than the per-caption-length cost.
