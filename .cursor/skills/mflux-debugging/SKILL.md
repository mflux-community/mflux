---
name: mflux-debugging
description: Debug MLX ports by comparing against a PyTorch/diffusers reference via exported tensors/images (export-then-compare).
---
# mflux debugging (MLX parity vs PyTorch/diffusers)

Use this skill when you are porting a model to MLX and need to **prove numerical parity** (or isolate where it diverges) versus a **PyTorch reference implementation** (often from diffusers).

This skill defaults to **export-then-compare**:
- Run the reference once and **export deterministic artifacts** (tensors + optional images).
- Load those artifacts in MLX and compare with clear thresholds.

## When to Use

- You suspect a port mismatch (wrong shapes/layout, RoPE, scheduler math, dtype casting, etc).
- You want a repeatable workflow to narrow down the first layer/block where outputs diverge.
- You need evidence of correctness before refactoring (see `mflux-model-porting`).

## Ground Rules (repo norms)

- Use `uv` to run Python: `uv run python -m ...`
- If you run pytest, preserve outputs: `MFLUX_PRESERVE_TEST_OUTPUT=1` (see `mflux-testing` and the Makefile test targets).
- Do **not** update or replace reference (“golden”) images unless explicitly asked.
- Debug artifacts (tensor dumps) should live in a local folder and **must not be committed** unless explicitly asked.
- If you need the broader porting workflow (milestones, ordering, when to refactor), follow `mflux-model-porting`.
- **RNG warning**: PyTorch and MLX RNGs are different. Matching the same integer `seed` is not enough for parity—export the *exact* initial noise/latents from the reference and load them in MLX.
- Practical setup: the PyTorch reference repo (often `diffusers/`) and `mflux/` are frequently **next to each other** on disk (e.g. both on your Desktop). Use absolute paths when in doubt.

## Default Workflow (export-then-compare)

### Preferred workflow: two tiny scripts + inline dumps

For day-to-day debugging, prefer a **minimal paired repro**:
- One simple script in the reference repo (often `diffusers/`), e.g. `diffusers/flux2_klein_edit_debug.py`
- One simple script in `mflux/`, e.g. `mflux/flux2_klein_edit_debug.py`

Keep them “boring”: hardcoded variables, no cli, no framework, and just a few `np.savez(...)` / `mx.save(...)` lines at the right spot.

The key trick for RNG parity:
- In the reference script, **compute latents once**, save them, then **pass them back into the pipeline** (`latents=...`) so the run definitely uses the dumped tensor.
- In rare cases where a tensor needs to be saved from within a loop, make sure its name reflects the loop conditions (e.g the 4th noise predcition in a 10 step loop etc).
- In the MLX script, **load that same latents file** and feed it into the MLX run (do not rely on matching integer seeds).

### 1) Pick a single deterministic repro

- Fix **seed(s)**, **prompt(s)**, **height/width**, **steps**, **guidance**, and any **input image** paths.
- Keep the first repro small if possible (fewer steps, smaller resolution) to iterate quickly.

### 2) Decide your checkpoints (what to dump)

Start coarse, then narrow:
- **VAE**: packed latents before decode; optional intermediate activations for one block if needed.
- **Transformer**: hidden states at entry/exit of the model, then per-block (or every N blocks), then inside attention/MLP.
- **Text encoder**: token ids + attention mask, embeddings output, then per-layer hidden states if needed.
- **Scheduler**: timesteps/sigmas/alphas and the predicted noise/velocity per step.

Tip: work “backwards from pixels” like `mflux-model-porting` suggests: validate VAE decode first with exported latents, then the diffusion/transformer loop, then text encoder.

### 3) Export artifacts from the PyTorch reference (no logic changes)

Create a run directory like:
- `./debug_artifacts/<run_id>/ref/`

Export with one of these patterns:
- **NumPy**: `np.savez(path, **tensors_as_numpy)`
- **PyTorch**: `torch.save(dict_of_tensors, path)`

### 4) Run the MLX side with the same inputs and compare

Create a matching run directory:
- `./debug_artifacts/<run_id>/mlx/`

Load and compare tensors. For each checkpoint, report:
- **Shape + dtype**
- **max_abs_diff**, **mean_abs_diff**
- **max_rel_diff** (guarding division by zero)
- Pass/fail with a clearly stated **rtol/atol**
- It is **more important to inspect actual tensor values** (e.g., first 10 elements) than rely on summary stats.
- Statistics can mislead; small-looking stats can hide systematic drift or sign flips.
- Prefer **runtime tensor dumps** over code reading; code can use different conventions yet still represent the same math.

Suggested tolerance starting points (adjust per component):
- **fp32 comparisons**: `atol=1e-5`, `rtol=1e-5`
- **fp16/bf16 comparisons**: `atol=1e-2`, `rtol=1e-2`
- If comparing images: compare both (a) tensor space before final clamp and (b) saved `png` visually, since tiny numeric diffs can look identical.

If a checkpoint fails:
- Add an **earlier** checkpoint and repeat (binary search through the forward path).

## Common Causes of Divergence (high-signal checklist)

- **Layout mistakes**: NCHW vs NHWC, transposes around convs/attention, flatten/reshape ordering. Some operations like convolutions can have different conventions between libraries.
- **Broadcasting**: scale/shift vectors applied on the wrong axis (common in RoPE and modulation).
- **Dtype casting**: reference silently upcasts to fp32 for norm/softmax; MLX path stays in fp16.
- **RoPE details**: position ids, reshape order, whether cos/sin are broadcast over heads vs sequence.
- **Scheduler math**: timestep indexing, sigma/alpha definitions, and off-by-one step order.
- **Scheduler config**: compare sigma schedules directly.
- **Seed/RNG**: ensure you aren’t comparing stochastic paths (dropout, noise sampling) without controlling RNG.
- **Device dtype**: MPS `float16` can produce NaNs; prefer `bfloat16` for reference dumps if you see NaNs.
- **Do not use CPU** for comparisons; always keep reference runs on MPS to avoid misleading behavior.

## Artifact Hygiene

- Prefer `debug_artifacts/<run_id>/...` at repo root.
- Do not commit `debug_artifacts/` unless explicitly asked.
- If you convert the parity check into a test, follow the repo’s testing conventions and preserve outputs (see `mflux-testing`).
- Clean up old artifacts when they are no longer needed, only focus on the current problem and avoid confusion with older artifacts that are not relevant for the current task.

## Isolate transformer + VAE (inject reference initial latents)

When full end-to-end images differ between mflux and diffusers but you need to trust the **core model** (transformer, text encoder, VAE decode), export the reference **initial noise** and run mflux denoising from that tensor. If injected-latent mflux looks good while native-seed runs differ, the forward path is likely sound and any gap is probably RNG, scheduler, or optional reference-only components.

**Do not commit** these scripts or Desktop `.npy` files unless asked. Use inline `uv run python <<'PY' ... PY` heredocs in the reference repo and in `mflux/`.

### Step 1 — Export initial latents from diffusers

Run from the local `diffusers/` clone (often next to `mflux/` on Desktop). Match mflux settings: height, width, seed, dtype (`bfloat16` on MPS).

Load the reference pipeline the same way you would for a normal generation, but **disable optional subsystems** that mflux does not implement or download (check `model_index.json` vs mflux `get_download_patterns()` / weight definition). Pass `None` for unused components, set pipeline flags to skip them, and use `local_files_only=True` so missing weights surface immediately instead of downloading extras.

Then export noise with the same shape helper the pipeline uses internally:

```python
from diffusers.utils.torch_utils import randn_tensor

pipe = <ReferencePipeline>.from_pretrained(
    "<org/model>",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    # plus any kwargs to omit optional reference-only modules
)
latent_h = height // pipe.vae_scale_factor  # or read from pipeline/config
latent_w = width // pipe.vae_scale_factor
ch = pipe.transformer.config.in_channels    # name varies by architecture
generator = torch.Generator(device="mps").manual_seed(seed)
latents = randn_tensor(
    (1, ch, latent_h, latent_w),
    generator=generator,
    device="mps",
    dtype=torch.bfloat16,
)
np.save("/path/to/init_latent.npy", latents.detach().cpu().float().numpy())
```

### Step 2 — Denoise in mflux from the loaded latent

If `generate_image()` has no `latents=` argument, **inline the denoising loop** from the variant class instead of patching production code:

1. `latents = mx.array(np.load(path)).astype(mx.bfloat16)`
2. Text encode step from the variant (e.g. `_encode_prompts`, prompt cache, etc.)
3. Any position encoding / conditioning setup the variant does before the loop
4. Denoise loop: noise prediction + `config.scheduler.step(...)` over `config.time_steps`
5. VAE decode + `ImageUtil.to_image(...)` (or the variant’s decode helper) → save PNG

Also run one `generate_image(seed=...)` baseline on the same prompt for side-by-side.

### How to read results

| Comparison | What it tells you |
|---|---|
| mflux native seed vs itself (re-run) | Sampling is deterministic on this hardware |
| mflux + diffusers latent vs diffusers full run | Transformer + VAE + CFG path quality |
| mflux native seed vs diffusers full run | **Not** a parity test — different RNG (`mx.random` vs `torch.Generator`) and often different sigma schedules |
| mflux + diffusers latent vs mflux native seed | Large diff is **expected** if schedulers/sigmas differ, even with identical starting noise |

If **injected-latent mflux** looks good, the port’s core forward path is trustworthy. Golden CI tests should still lock **mflux-native** sampling (see `mflux-testing`), not diffusers pixel parity.

## diffusers reference runs (side-by-side images)

- mflux does **not** depend on diffusers; run reference scripts from a sibling `diffusers/` clone with `cd .../diffusers && uv run python`.
- Use `HF_HUB_OFFLINE=1` when weights are cached to catch missing components the reference pipeline expects but mflux omits from its download patterns.
- Compare at matched resolution, steps, guidance, seed, and precision when possible (note mflux `-q 8` vs diffusers bf16 is not apples-to-apples for speed or pixels).
- Save outputs to explicit paths (e.g. Desktop) and report wall time with `/usr/bin/time -p`.

## See Also

- `mflux-model-porting`: correctness-first workflow (validate components and lock behavior before refactor).
- `mflux-testing`: how to run tests safely and handle image outputs/goldens.

