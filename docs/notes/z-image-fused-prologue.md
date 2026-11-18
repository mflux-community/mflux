# Fusing the Z-Image attention prologue

## Why this note exists

A measured breakdown of one Z-Image DiT block puts the linear layers at about 81% of block time and
attention at about 17%, at S=4224. Everything that is neither of those is a few percent. That last
term is the whole budget for this change. It is also the least reliable number in the measurement. It is
computed by subtraction, and it spanned 1.4% to 6.8% across runs of configurations that should not
differ. Quote it as an order of magnitude. Do not quote it as 4.7%.

Figures from 2026-11-18 on an M4 Max, one compiled block at S=4224, random bf16 weights. The harness
lives at `spikes/zimage_dit/` on branch `investigation/z-image-dit-nonmatmul`, PR #1001, not on
`main`. That branch's README carries the result tables and their provenance. Rebuild it from this
recipe if you need the split elsewhere. Use `dim=3840`, 30 heads, head_dim 128 and hidden 10240. Time
one `mx.compile`d `ZImageTransformerBlock`. Then time the same block reduced to its seven matmuls, and
the SDPA call alone.

## What already exists

`src/mflux/models/qwen21/model/qwen21_transformer/qwen21_fused_kernels.py:84` builds one Metal
kernel through `mx.fast.metal_kernel`. `fused_qk_norm_rope` (`:108`) does three jobs in one pass:
RMSNorm on Q and K, RoPE in float32, and the change from `(B, L, H, D)` layout to `(B, H, L, D)`
layout. It reads the projection, norm-weight and rotary-table views through their original strides.

The gate `fused_qk_norm_rope_available` (`:99`) accepts any `head_dim` where `head_dim % 64 == 0`
and `head_dim <= 512` (`:103`). Z-Image uses head_dim 128, so the gate already passes.

## What Z-Image does today

`attention.py:61-79` reshapes to `(B, S, H, D)`, applies `norm_q` and `norm_k`, calls
`apply_rotary_emb`, then transposes Q, K and V. `apply_rotary_emb` (`:97-111`) runs about ten eager
operations and materialises a new tensor at `mx.stack` (`:107`). So the norm, the rotation and the
layout change each cost their own pass.

## The pairing convention already matches

An earlier analysis claimed Qwen rotates half the head while Z-Image rotates interleaved pairs.
That claim was wrong. Both rotate adjacent pairs.

- Kernel: `out[c] = nqa * cosine - nqb * sine` and `out[c + 1] = nqa * sine + nqb * cosine`, where
  `p = c / 2` (`qwen21_fused_kernels.py:58,67-68`).
- Z-Image: reshape to `(..., head_dim // 2, 2)`, take `x[..., 0]` and `x[..., 1]`, and apply the
  same two formulas (`attention.py:100-108`).

No pairing change is needed. This makes the port cheaper than it first looked.

## Three real gaps to close

1. **Epsilon.** The kernel hardcodes `1e-6` (`qwen21_fused_kernels.py:17`). The Z-Image QK norms use
   `1e-5` (`attention.py:18`, applied at `:34-35`). Pass epsilon in as an argument.
2. **Table layout.** The kernel takes separate `rope_cos` and `rope_sin` arrays of shape
   `(L, D/2)`. Z-Image holds one `freqs_cis` of shape `(S, D/2, 2)` (`rope_embedder.py:37-40`). The
   two slice views have the right shape, and the kernel reads by strides, so this probably works.
   Verify it rather than assume it.
3. **The `--float32` path.** Z-Image keeps the RoPE output in float32 when `float32` is set
   (`attention.py:111`), and otherwise casts it back to the dtype it received
   (`attention.py:98` and `:111`). The kernel's output dtype is not the input dtype.
   `TQ` and `TK` bind to `mx.result_type(q_flat, weight_q)` and `mx.result_type(k_flat, weight_k)`,
   and those same two values are the `output_dtypes`
   (`qwen21_fused_kernels.py:141-148`). The docstring at `:122` says it plainly: "each native
   RMSNorm's promoted dtype". So bf16 activations against float32 norm weights
   come back float32, not bf16. A port must reproduce the promotion rather than assume it, and must
   check which side of it Z-Image's own `set_dtype` puts the QK norm weights on.

## Things to keep when porting

- **Escape hatch.** `MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE=1` selects the composed path
  (`qwen21_fused_kernels.py:101`). Give the Z-Image kernel the same kind of switch.
- **Test to copy.** `tests/test_qwen21_perf.py:16`, `TestFusedQkNormRopeKernel`. It already
  parametrises head_dim 128.
- **Rounding intent.** The kernel deliberately reproduces MLX native RMSNorm promotion and rounding,
  and it turns off FP contraction for the rotary terms (`:53`). Keep both, or the fused path stops
  matching the composed path bit for bit.

## Expected gain

RoPE plus the norm plus the layout change is a subset of that few-percent non-matmul term, so the
ceiling is the whole of that term. Treat 4.7% as a loose upper bound rather than a measurement, and
expect 1 to 2%. Measure the fused path against the composed path at S=4224 before and after. Keep the
bitwise comparison in the test. Repeat each timing enough times to clear the run-to-run band, which is
about plus or minus 1.5% of block time for this method.
