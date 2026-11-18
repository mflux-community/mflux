Written for: a new AI Code session picking up Stage 3 of the Z-Image ANE spike.

# Hand-off: Z-Image ANE spike, Stage 3

## State
- **Repo and branch:** `/Users/ianscrivener/MFlux/mflux-upstream`, branch `experiment/zit-on-ane`, pushed to the `ianscrivener` remote (fork) at `3154afeb`. No PR is open.
- **Spike code:** `spikes/ane_matmul/`. Read `REPORT.md` first.
- **Left alone:** `spikes/zimage_dit/` is untracked and is Agent 2's. Do not delete or commit it without asking them.
- **Environment:** MLX is 0.32.4 in the project `.venv`, and coremltools 9.0 is installed.
- **Hooks:** ruff, typos, ty and `uv sync --locked` all pass.
  - Lint exceptions are scoped to `spikes/**`.
  - The typos allowance for `ane`/`ANE` is global.

## Results (M4 Max, Z-Image Turbo q8, 1024², 4128 tokens)
- **Best form:** baked int8 weights in a 1x1 Conv2d, channels-first, with weight-scale constants.
  - For w2 (10240 inner dimension), split K into 4 chunks. Each chunk needs its own weight constant, not a slice of one constant.
  - Dynamic weights as inputs (V1) are slower than the GPU. Do not use them.
- **Layer times (ms):**

  | Layer | MLX bf16 | MLX q8 | ANE W8A16 | ANE W8A8 per-token |
  |---|---|---|---|---|
  | 3840×3840 | 10.5 | 11.9 | 9.1 | 7.1 |
  | 10240×3840 | 27.6 | 31.3 | 22.8 | 19.7 |
  | 3840×10240 (w2) | 28.2 | 32.4 | 25.3 | 17.5 |

- **Sum over 30 blocks, linear layers only:** bf16 3.76 s, q8 4.27 s, W8A16 3.22 s, W8A8 2.56 s.
- **Concurrency (GPU bf16 + ANE W8A16, w1 shape):** the best split is 50:50, at 1.97× the GPU alone. The GPU half does not slow down when the ANE runs alongside it.
- **Stage 2, block 15 w2:** a 50:50 split in mflux runs at about 19.5–20 ms against 30.6 ms GPU-only (about 1.55×), cosine 0.999994. Images match with the flag on (PSNR 29.5 to 43.5 dB). Step time did not change, because one layer is about 0.2% of a step.
- **Accuracy (block 15, worst cosine):** W8A16 0.99999, W8A8 0.9996 to 0.9998 (max diff about 4% of peak on to_q).

## Gotchas
- **fp16 overflow:** w2 outputs reach about 80,000, above the fp16 maximum of 65,504. Divide the input by 16 before the ANE and multiply the output by 16 on the GPU. Scale in bf16 before the cast to fp16.
- **Stream ordering:** without a data dependency that makes the ANE launch first, the GPU and ANE halves run one after the other. `ane_split.py` has an ordering hack, and `MFLUX_ANE_ORDER_HACK=0` turns it off. The join must be the only consumer of the ANE result.
- **nanobind ABI:** it must match MLX.
  - MLX 0.32.0–0.32.3 need nanobind 2.13.x or 2.14.x (ABI v20).
  - MLX 0.32.4 needs nanobind 3.1.x (ABI v22). CMake checks this.
  - The nanobind install used for the build is in the old session's scratchpad and may be gone. To rebuild:
    1. Install nanobind into a scratch folder with `uv pip install --target <dir> nanobind==3.1.0`.
    2. Run `PYTHONPATH=<dir> cmake -S spikes/ane_matmul/ane_linear -B <build> -DPython_EXECUTABLE=<repo>/.venv/bin/python -DMLX_DIR=<repo>/.venv/lib/python3.13/site-packages/mlx/share/cmake/MLX`.
    3. Run `cmake --build <build>`, then copy the `.so` into `ane_linear/`.
- **Objective-C:** the bridge needs ARC (`-fobjc-arc`); CMake sets it.
- **Compiled predict:** mflux wraps the DiT predict in `mx.compile`. Capture scripts disable it with `mx.compile` set to identity. The patched run used the compiled path on M4. Other chips are untested.
- **Fixed token count:** the Core ML model is fixed at 4224 tokens (1024²). Other resolutions need other models.
- **Scratchpad cleanup:** it deletes files. Real activations are in `/Users/ianscrivener/ane_spike_acts/`, with e2e images in the `e2e/` subfolder there. The Core ML cache is in `ane_linear/cache/` and is git-ignored.

## Local rules that still apply
- Run anything on the GPU or ANE under `spikes/ane_matmul/watchdog.sh`. It kills the job if memory goes above 97%.
- Ask before any real-weight or generation run, and tell the user to close apps first.
- Local models:
  - **q8:** `/Volumes/Extreme2Tb/MFlux-Models/Z-Image-Turbo-mflux-q8`.
  - **bf16:** the Hugging Face cache, `Tongyi-MAI/Z-Image-Turbo`. Do not download.
  - **Z-Image Base:** no tests on it.
- Written text must pass the STE lint. Run `~/.claude/skills/asd-ste100-skill/scripts/ste-lint.py` on it.
- Ask "Do we run CodeRabbit locally first?" before any push.

## Stage 3 work, in order
1. **Decide the path.** W8A16 is the safe path. W8A8 is the fast path, but its accuracy margin is thin and error over 30 blocks is not measured. Recommendation: build W8A16 first, then test W8A8 at image level.
2. **Range audit:** capture input and output ranges for every linear layer across all steps of one 1024² run. Flag any layer above about 30,000 and keep those on the GPU.
3. **All layers:** generate models for all layers. Use the 4-chunk K-split for w2-shaped layers and the 50:50 split.
4. **Token counts:** choose a fixed set of token counts, or pad to 4224.
5. **W8A8 only:** time the GPU quantization step, which was not measured.
6. **Full-image quality:** run PSNR and a visual check over several prompts, and check memory.
7. **Open items:**
   - The standalone ANE vs bf16 criterion (≥1.5×) fails for W8A16 and nearly passes for W8A8.
   - Per-call overhead is not measured on its own.

