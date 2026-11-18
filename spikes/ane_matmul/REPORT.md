# Z-Image ANE matmul spike: report

Machine: M4 Max, 36 GB, macOS 27.0.1. Model: Z-Image Turbo q8, 1024 x 1024, 8 steps.

## Stage 1 results (4128 tokens, median of 3 runs, ms)

| Layer | MLX bf16 | MLX q8 | ANE W8A16 | ANE W8A8 per-token |
|---|---|---|---|---|
| 3840 x 3840 | 10.52 | 11.87 | 9.10 | 7.10 |
| 10240 x 3840 | 27.60 | 31.29 | 22.83 | 19.70 |
| 3840 x 10240 (w2) | 28.17 | 32.41 | 25.30 | 17.46 |

- Best form: baked int8 weights in a 1x1 Conv2d. The w2 layer needs the inner dimension split in 4 chunks with one weight constant for each chunk.
- Dynamic weights as inputs (V1) are slower than the GPU. Do not use them.
- All ops show as ANE in the Core ML compute plan.

## Concurrency (GPU bf16 + ANE W8A16, 4224 x 10240 x 3840)

| GPU : ANE columns | Layer time | Speed-up over GPU alone |
|---|---|---|
| 100 : 0 | 28.5 ms | 1.00x |
| 60 : 40 | 17.2 ms | 1.65x |
| 50 : 50 | 14.4 ms | 1.97x |
| 40 : 60 | 15.4 ms | 1.85x |

Best split: 50 : 50.

## Accuracy against fp32 (block 15, 3 prompts, worst cosine)

| Layer | std q8 | W8A16 | W8A8 |
|---|---|---|---|
| to_q | 0.999993 | 0.999990 | 0.99965 |
| w1 | 0.999993 | 0.999988 | 0.99961 |
| w2 | 0.999996 | 0.999993 | 0.99976 |

- The w2 output peaks near 80,000. This is more than the fp16 maximum of 65,504. Divide the input by 16 before the ANE and multiply the output by 16 on the GPU.

## Stage 2 (single layer: block 15, w2)

- `ane_linear/`: MLX custom primitive, Core ML bridge, nanobind module. The input and output buffers are wrapped without a copy. Core ML can still copy inside.
- `ane_linear/ane_split.py`: column split, opt-in with `MFLUX_ANE_LINEAR=1`. Default mflux behavior is unchanged.
- Layer, 50:50 split, 4128 tokens: 21.3 ms. GPU only (q8): 32.2 ms. Speed-up: 1.51x. Cosine 0.999994.
- Full image, 3 prompts, flag on against flag off: PSNR 40.7, 29.5 and 43.5 dB. The images look the same.
- Step time did not change in a measurable way: one layer of 30 blocks is about 0.2 % of a step.

## Go / no-go

| Criterion | Result |
|---|---|
| ANE vs GPU bf16, standalone, at least 1.5x | W8A16: 1.11 to 1.21x (fail). W8A8: 1.40 to 1.61x (4128 tokens) |
| GPU + ANE at least 1.3x the GPU alone | 1.51x for the single layer in mflux. 1.97x in the synthetic test. Pass |
| Per-call overhead at most 5 % | Not measured separately. In-model transposes cost 3 to 4 % |
| Accuracy | Pass for W8A16. W8A8 passes the cosine limit by a small margin |

## Risks

- The spike patch ran with `mx.compile` on M4 (patch effective, images changed). Other chips are not tested.
- The MLX ordering of GPU and CPU streams matters. Without a data dependency that forces the ANE launch first, the two halves run one after the other (35 ms instead of 21 ms).
- The nanobind ABI must match MLX (version 2.13.0 for MLX 0.32.0).
- Token count is fixed at 4224 in the Core ML model. Other resolutions need other models.
- W8A8 error over 30 blocks is not measured.
- The W8A8 model applies the per-token scale after the matmul. The partial sums can exceed fp16 before that scale if int8 weights are unit scale. With real weight scales (about 0.0005 to 0.01) a worst-case test stays finite. Real activations passed the accuracy test.
- fp16 range audit over all layers and steps is not done.

## Stage 3 estimate

About 2 to 3 weeks. The work items are:

- Generate models for all layers.
- Fix a set of token counts.
- Run the range audit.
- Do W8A8 quantization on the GPU.
- Check full-image quality.
- Tune memory.
