# Eager GPU+ANE orchestration: results (Z-Image Turbo, 1024x1024, 8 steps)

Question: the row-split MLP overlaps the ANE with the GPU only when the ANE call runs on a Python thread, which
`mx.compile` forbids. Can each block be compiled on its own, with plain Python launching the threaded ANE MLP between
the compiled calls? **Yes.** Per-block compile costs nothing (B/A 1.001), and the threaded split gives 1.23x to 1.33x
at image level.

Scripts (this directory): `eager_bench.py` (block-level A/B/C/D), `eager_image_child.py` and `eager_image_orch.py`
(image arms, one process per arm), `timeline_probe.py`, `pre_breakdown.py`, `fused_accuracy.py`,
`trace_capture.py`, `primitive_capture.py`. Logs: `results_eager/`.

Layer schedule in arm C (medians, ms from layer entry): pre graph done 77.5, ANE start 77.55, GPU rows done 123.2,
ANE end 125.6, epilogue done 127.2. The ANE starts 0.05 ms after the pre graph; the earlier primitive form started it
30 to 96 ms late. A layer is about 127 ms in C against about 160 ms in A.

## Block level (2 refiners + 30 main layers, real bf16 weights, S=4224, share 0.50, 5 repeats)

| Arm | Step ms | Ratio |
| --- | --- | --- |
| A whole-chain compile | 5118.5 | n/a |
| B per-block compile | 5113.2 | 1.001 vs A |
| C B + threaded ANE MLP | 4074.8 | 1.256 vs A |
| D C with MLX_METAL_FAST_SYNCH=1 | 4075.6 | no effect |

## Image level (2 prompts, seed 1, text encoder dropped after encoding, one process per arm, BENCH.lock, nice 0)

| Run | Arms | Image time s (median) | Ratio |
| --- | --- | --- | --- |
| v2 (3 reps) | A, C | 41.28, 33.45 | C/A 1.234 |
| v3 (3 reps, after reboot) | A, C, A8, C8 | 41.04, 33.36, 44.31, 34.39 | C/A 1.230, C8/A8 1.288 |
| v4 (3 reps, after reboot, fused QK-norm+RoPE) | A, A+F, C+F, A8, C8+F | 41.13, 39.84, 31.98, 44.23, 33.16 | C+F/A 1.286, C+F/A+F 1.246, C8+F/A8 1.334 |

Without dropping the text encoder (v1) arm C paged 3 to 6 GB of swap and took 83 to 155 s: invalid. Stock q8 is 8%
slower than stock bf16 on this Mac. Swap did not grow inside any timed window in v2 to v4; it grew between windows
from model loads. Image difference, same seed: C vs A 35.7 dB (prompt A) and 27.6 dB (prompt B); C8 vs A8 38.5 and
24.9 dB. All images were approved by eye. The fused kernel is bit-identical to the composed ops.

## Not solved

C is a spike, not product code: no per-block eager path in mflux, ANE int8 built at load (about 3.5 GB host), no
low-RAM path, peak RSS up to 12.6 GB (C8). The Stage 3b file `src/mflux/models/z_image/ane/z_image_ane.py` is frozen,
and its open review findings are in `MAILBOX/coderabbit-eager-report.md` (outside this repo).
