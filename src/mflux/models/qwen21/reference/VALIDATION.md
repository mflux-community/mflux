# Qwen-Image-2.1 validation

Local verification on Apple Silicon with 128 GiB unified memory, 2026-09-21. These results describe the tested checkpoint and inputs; they are not a quality benchmark or a guarantee of PyTorch pixel parity.

- Checkpoint: `Qwen/Qwen-Image-2.1`, revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`. All seven weight files were checked against the Hugging Face LFS SHA-256 hashes (33,115,613,408 bytes).
- Reference Diffusers: `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`.
- Python 3.14, MLX 0.32.0, Torch 2.13.0, Transformers 5.15.0.
- Initial inference evidence was collected on `feat/qwen-image-2.1` with upstream `d9ee291` merged in. The PR packages the same reference implementation under `qwen21/reference` on upstream `8c00dab`, after #736 introduced a separate text-only implementation. It reuses that model registration and adds `mflux-generate-qwen-2.1-edit`; the original command, save routing, and golden images remain unchanged. PR-branch checks and a saved-checkpoint comparison are recorded below.

## Automated checks

| Check | Result |
| --- | --- |
| `just test-fast` | 1504 passed, 10 skipped, 229 deselected |
| Model tests with pinned Diffusers installed | 35 passed, including 13 numerical reference cases |
| `just lint` | Passed using the cached pinned Ruff (`UV_OFFLINE=1`) |
| `just typecheck` | Passed |
| `just build` | Wheel and sdist built successfully |
| Existing golden images | Unchanged |

The regular fast run skips seven optional Diffusers comparisons when Diffusers is absent. The separate 35-test run installs the pinned reference and executes them. A temporary macOS native-library loading stall delayed the fast run; it subsequently completed. A direct `PYTHONPATH` injection of Diffusers failed on an incompatible Hub dependency; the documented `uv run --with` installation resolved the reference environment and passed.

## Real weights and numerical comparisons

The checkpoint's tensor keys and shapes match all three components: VAE 238, transformer 297, text/vision encoder 748. Small-model tests use identical reference weights and inputs, including changing target latents between cached steps, padding masks, adjacent reference images, multimodal position IDs, DeepStack injection, and fp32/bf16 text encoding.

| Real component | Inputs | Observed difference |
| --- | --- | --- |
| VAE fp32 encode | 256-pixel RGBA fixture | Maximum absolute error 1.19e-4; mean 2.78e-6 |
| VAE fp32 decode | Identical latents | Maximum absolute error 1.76e-5; mean 5.27e-7 |
| Full 32-layer transformer fp32 | Identical random inputs, text-only / one reference, prefill / cached | Relative L2 1.76e-5 to 4.52e-5; maximum absolute error 2.79e-4 |
| Text/vision encoder fp32 | Text-only / one / two 256-pixel images | Relative L2 6.53e-6 / 1.21e-4 / 9.95e-5 |

VAE comparisons pass `atol=1e-4, rtol=1e-4`. The full real transformer exceeds that strict elementwise threshold at 2/1024 text-only prefill elements and 22/1024 image-conditioned prefill elements; both cached cases pass it. The comparison records this rather than relaxing the small-model test tolerance.

BF16 encoder differences are substantially larger: MLX versus Torch relative L2 is approximately 0.026, 0.152, and 0.216 for zero, one, and two images. Against Torch fp32, Torch bf16 itself differs by 0.054, 0.140, and 0.217; MLX bf16 differs by 0.047, 0.167, and 0.227. These measurements establish sensitivity to reduced precision, not equality of the resulting denoising trajectories. These standalone encoder measurements do not imply pixel parity; see the controlled denoising comparison below.

## Real generation and reload

All cases use seed 42, guidance 1, and prefix KV caching. Times include competing local workloads and should not be used as performance benchmarks. Except where stated, outputs use 40 steps.

| Case | Settings | Result |
| --- | --- | --- |
| Text-to-image | Dense bf16, 1024x1024 | Coherent watercolor red panda reading a book; about 591 seconds denoising, 31.9 GB peak MLX memory |
| Single-reference editing | Dense bf16 and 8-bit, 512x512 output/reference budget | Both produce snow and preserve the red panda and book; 8-bit run about 122 seconds denoising, 19.5 GB peak |
| Two-reference editing | 8-bit, 512x512 output/reference budget | Both red panda and puffin appear in the requested watercolor forest; about 365 seconds denoising with concurrent inference, 19.5 GB peak |
| Two-reference clothing edit | 8-bit, 896x1152 output, 1024 reference budget, **25 steps**, seed 1070478148268574 | Person, pose, and background retained; dark pullover replaced with the second reference's light blue denim shirt; 446 seconds total, 31.5 GB peak |
| Transparent PNG | 8-bit, 1024x1024 | Red panda sticker; alpha 0–255, 67.3% of pixels below 32 and 32.2% above 223; 503 seconds total |
| 2048 generation | 8-bit, 2048x2048, VAE tiling, **4 steps** | Execution/shape smoke test passed, about 288 seconds; full-step quality unverified |
| Quantized export/reload | 8-bit, 512x512, 8 steps | 17.8 GB local checkpoint; before/after reload pixels exactly equal |

Single- and two-reference edits passed visual checks at 512x512, and a two-reference clothing edit passed at 896x1152. **The 1024x1024 panda edits did not pass visual acceptance.** Both dense bf16 and 8-bit single-reference snow edits retained the subject but showed excessive sharpening and did not clearly perform the requested seasonal change. The dense comparison completed 40 steps in about 706 seconds, confirming this observation is not specific to quantization. A two-reference 1024x1024 output also omitted the second subject. The follow-up controlled reference run below reproduces the same outcome with the official transformer, text encoder, and scheduler. This sample therefore does not establish an MLX-specific defect. The successful clothing edit does not resolve or replace those failing samples.

Local evidence is preserved under `debug_artifacts/qwen21/` (ignored by Git), including source hashes, comparison JSON/arrays, initial noise and denoising latents, PNGs with metadata, diagnostic scripts, and the exported checkpoint. `acceptance/COMPLETE` records execution and reload checks only; visual acceptance is described above. No generated images were installed as golden references.

## Follow-up: actual 1024 inputs

The follow-up compares the failing snow-edit prompt, reference image, and initial noise directly. Reference tensors and scripts are retained in `debug_artifacts/qwen21/actual-conditioning/` and adjacent files.

- Actual prompt encoding: fp32 relative L2 over **text positions** is 8.11e-6 at 512 and 1.16e-5 at 1024. BF16 differences at those positions are 2.94% and 3.60%. The much larger errors over image positions are not a useful standalone quality indicator because the transformer replaces those positions with VAE features.
- Actual 1024 transformer prefill: identical embeddings, reference latents, and noise give fp32 relative L2 7.16e-6 (maximum absolute error 0.00135), and bf16 relative L2 0.0146. Selected activations from all 32 layers were exported.
- Actual 1024 decode: identical final latents give maximum pixel-space error 3.87e-5, mean 6.80e-7, relative L2 1.51e-6.

A large apparent VAE encoding mismatch was isolated to **PyTorch 2.13.0 MPS temporal padding**. Padding an all-ones tensor of shape `(1, 96, 1, 256, 256)` with `(0, 0, 0, 0, 1, 0)` returned zeros for both frames; the same operation at side 128 was correct. The NumPy result and MLX shortcut agree exactly. The reference diagnostic now prepends an explicit zero tensor with `torch.cat`, preserving the original mathematical operation; no production VAE change was made to imitate the erroneous MPS result. With that workaround, sampled 1024 encoder output has relative L2 1.67e-6. The reproduction is `debug_artifacts/qwen21/mps_pad_probe.py`.

The reference tests now execute on MPS. Two independent constant-image tests protect first-frame downsampling at the sizes on either side of the observed reference failure. The focused run passes all 35 cases. The subsequent 40-step reference run completed in about 650 seconds. It uses the pinned official Qwen3-VL embeddings, bf16 transformer, prefix cache, and FlowMatch Euler scheduler. The same initial noise and verified fp32 VAE condition latents are injected into both paths; both final latent tensors are decoded by the same MLX VAE, whose 1024 decode was checked independently above. This isolates the denoising implementation rather than pretending two different RNGs provide the same input.

The reference also retains the blossoms, oversharpens the subject, and fails to establish a convincing winter scene. Its RGBA output versus the dense MLX baseline has mean absolute pixel difference **0.388 / 255**, maximum difference 46, and **PSNR 49.91 dB**. The outcome is shared by the reference path for these inputs, not evidence of a defect specific to the MLX port. `actual-conditioning/reference-edit.png` and `image-metrics.json` preserve the result.

As a separate precision check, the corrected MPS bf16 VAE condition differs from the fp32 condition by relative L2 0.00677. The measured image comparison uses fp32 VAE conditioning consistently; it is not claimed to be a complete untouched bf16 Diffusers pipeline run.

Two additional 1024-pixel samples with explicit descriptions also failed visual acceptance. The winter prompt explicitly requested snow-covered bare branches and removal of blossoms; the two-reference prompt specified the panda on the left and the puffin on the right. Both retained the panda-and-blossoms composition and missed the requested changes. They used the saved 8-bit checkpoint, 40 steps, seed 42, and guidance 1; reference budgets were 1024 and 512 respectively. Runs took 612 and 603 seconds with approximately 31.7 GB peak MLX memory. Outputs and metadata are in `explicit-edit/`. Prompt expansion did not resolve the observed limitation.

The current Hugging Face head, `790c92633540aa0cb11d9abf19eb46d861714758`, was checked on 2026-09-21. Its weights and component configurations are unchanged from the tested revision; the changes are repository documentation/assets. The current Diffusers pipeline source also matches the pinned pipeline. A source comparison with ComfyUI's native `qwen_image21` implementation found the same prompt template, pre-final-normalization text features, reference insertion order, zero-timestep prefix, and centered image positions for the even latent dimensions supported here. This source review is not a ComfyUI runtime parity test.

## Follow-up: official workflow inputs

The [ComfyUI Qwen-Image-2.1 editing workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json) supplies a portrait, a light blue denim shirt, and a prompt requesting that the person wear the shirt while preserving identity, pose, and background. Those two source images and the workflow's prompt were run through the native MLX API using the saved 8-bit checkpoint, 25 Euler steps, guidance 1, seed 1070478148268574, 896x1152 output, and a 1024 reference budget. Both reference images retain their original 896x1152 dimensions after budget rounding.

The output passes the stated visual check: the person keeps the original pose and background, and wears a light blue denim shirt with the reference collar, buttons, and chest pocket. Total generation time was 445.6 seconds; peak MLX allocation was 31.5 GB. Inputs are in `comfy-reference/`; the result, metadata, and timing are in `workflow-edit/denim.png` and adjacent files. This validates a real megapixel two-reference edit. It is not a pixel comparison with ComfyUI: the run uses mflux's RNG and the pinned Diffusers-compatible scheduler, rather than ComfyUI's sampler implementation.

The port therefore has successful real-weight text generation, transparent output, single-reference editing, multi-reference editing, and quantized save/reload evidence. The panda examples remain input-specific instruction-following failures with an unresolved cause; the comparisons do not establish that all square edits fail or that megapixel editing is generally broken. Full-step 2048 visual quality remains unverified.

## PR integration checks

The PR branch `feat/qwen-image-2.1-reference-edit` is based directly on upstream `8c00dab`; unrelated local history and staged documentation edits are excluded. `mflux-capabilities` discovers the new command with full parser coverage and conditional scheduler/negative-prompt declarations; the model inventory exports all 36 registered models.

- CI default selector (`not slow and not high_memory_requirement`): 1667 passed, 10 skipped, 63 deselected.
- Fast selector: 1500 passed, 10 skipped, 230 deselected.
- Focused tests with pinned Diffusers: 35 passed on MPS.
- Lint, formatting, type checking, and wheel/sdist build: passed.
- Installed `mflux-generate-qwen-2.1-edit` with the saved 8-bit checkpoint and original 512-pixel, eight-step metadata: output is pixel-identical to the pre-migration result.
- Installed editing CLI, 512-pixel snow edit, 40 steps, seed 42: output is also pixel-identical to the successful pre-migration reference edit.

The independent worktree reuses the already installed dependency packages through a local site-packages path and installs its own editable mflux distribution. A fresh offline locked sync could not find the public-PyPI Torch wheel URL in cache, so the fast selector was run with `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run --no-sync python -m pytest -m fast` instead of the recipe's mandatory sync. No dependency versions or lockfile were changed.
