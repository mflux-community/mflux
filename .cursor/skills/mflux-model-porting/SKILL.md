---
name: mflux-model-porting
description: Port ML models into mflux/MLX with correctness-first validation, then refactor toward mflux style.
---
# mflux model porting

## Goal
Provide a repeatable, MLX-focused workflow for porting ML models (typically from diffusers repo located near mflux repo in the system) into mflux with correctness first, then refactor to mflux style.

## Principles
- Match the reference implementation first; prove correctness before cleanup.
- Lock correctness with deterministic tests before refactoring.
- During the initial port, avoid premature performance work (e.g., `mx.compile`, kernel fusion tweaks, scheduler micro-optimizations); add optimizations only after correctness is locked.
- Refactor toward shared components and clean APIs once tests are green.
- PyTorch and MLX RNGs are different; for strict parity checks, export the *exact* initial noise/latents from the reference and load them in MLX instead of relying on matching integer seeds.

## Workflow (checklist)
1. **Scope and parity**
   - Define target parity (outputs, speed, memory) and acceptable tolerances.
   - Identify reference files, configs, and checkpoints to mirror.
   - Draft a Cursor plan for the port and review it before starting implementation.
2. **Port fast to reference**
   - Add the model package skeleton and a variant class + initializer.
   - Follow standard mflux initializer/weight-loading style; review recent ports like `z_image_turbo` and `flux2_klein` for structure and naming.
   - Wire weight definitions/mappings early so loading is exercised (implement quantization in the initializer, but skip it during early runs).
   - Keep the first implementation simple and explicit; defer `mx.compile` and other speed-focused changes until deterministic parity is passing.
   - When defining explicit weight mappings, inspect actual tensor values from the model in the Hugging Face cache to confirm names and shapes.
   - Add a minimal hardcoded runner for quick iteration (two tiny scripts: one in the reference repo, one in mflux), seeded with diffusers-style defaults (e.g., 1024×1024, default prompt).
   - Add lightweight shape checks close to the code paths.
   - Use `mx.save`/`mx.load` at critical points; it is OK to add these to the reference (without changing logic) to export latents.
3. **Port order (work backwards from image)**
   - Typical image generation flow: `prompt → text_encoder → transformer_loop → VAE → image`.
   - For porting, invert the order so you can validate pixel space early.
   - Start with VAE decode/encode to validate output images quickly:
     - Export packed latents from the reference just before VAE decode.
     - Load latents inline and decode to an image for visual inspection.
     - Run an encode→decode roundtrip to sanity check reconstruction; a good-looking image reconstruction increases confidence in the implementation.
     - Expect small numeric diffs in tensor values; when it is not clear from the numbers alone, always generate images and rely on human visual inspection to judge whether the match is acceptable.
   - Then port the transformer loop and its schedulers with intermediate latent checks.
     - If the reference uses a novel scheduler, port it; otherwise, reuse the existing mflux scheduler.
   - Finish with the text encoder and tokenizer details.
   - After each major component is validated (e.g., VAE, transformer, text encoder), commit with a clear milestone message like "VAE done" to preserve progress.
    - Once the full port is working, remove any loaded tensors or debug artifacts so no traces remain.
4. **Deterministic validation**
   - Create a deterministic MLX test (image or tensor) that locks the output.
   - Run tests via `MFLUX_PRESERVE_TEST_OUTPUT=1 uv run <test command>`.
   - If MLX OOMs on sensible inputs (e.g., 1024×1024), assume a likely porting mistake and re-check shapes or memory-heavy ops.
5. **Post-test refactor (explicit step)**
   - Review commits after the first deterministic test to capture refactoring preferences.
   - Consolidate shared components into common modules.
   - Remove debug paths and one-off schedulers once validated.
   - Move configuration defaults into standard config/scheduler paths.
    - Simplify and decompose large files into focused modules once behavior is locked.
    - Prefer shared scheduler implementations when they already exist in mflux.
    - Ensure CLIs register callbacks via `CallbackManager.register_callbacks(...)` so shared features like `--stepwise-image-output-dir` work; pass a `latent_creator` that supports `unpack_latents(...)`.
    - Keep running the deterministic image test during refactors to avoid regressions.
   - Align the variant class with recent ports (`flux2_klein`, `z_image`): `prompt_cache`, merged `_predict`, RoPE setup inside predict path, `_decode_latents` helper, no verbose comments/docstrings (see repo `RULE.md`).
   - Strip dead scaffolding (e.g. unused gradient-checkpointing flags) once training/inference paths are stable.
6. **Pre-merge polish (after core port works)**
   - **diffusers sanity check**: run matched mflux + diffusers generations; use `mflux-debugging` latent injection if outputs disagree but you need to validate transformer/VAE.
   - **Golden tests**: pick prompt/seed/settings that are stable on target CI hardware; update reference PNGs only after explicit approval (see `mflux-testing`).
   - **img2img**: verify latent packing/normalization on the img2img path matches txt2img and training (especially when reusing a shared VAE from another model family). `pack_latents` must accept the 5D `(B, C, 1, H, W)` tensor that tiled VAE encode (`vae_encode_tiled`) returns — squeeze the singleton temporal axis first, as `flux2`/`fibo`/`z_image` latent creators do; a 4D-only unpack (or a bare passthrough `pack_latents`) breaks tiled img2img. This is reachable via `--low-ram`: `MemorySaver` sets `tiling_config = TilingConfig()` (`vae_encode_tiled=True`), so always test img2img with `--low-ram`, not just the default path (which is safe only because `VAEUtil.encode` squeezes 5D→4D when tiling is off).
   - **Cross-model touch points**: list every file outside `models/<your_model>/`; justify shared changes (`memory_saver` tiling guard, shared VAE `tiling_config`, training `runner` wiring). Drop unrelated edits (e.g. personal `.gitignore` entries).
   - **README**: follow an existing model README structure (e.g. Flux2): hero image, turbo + base examples, feature section (img2img), disk-size warning, Notes, Training. Measure on-disk sizes with `du` on HF cache and/or `mflux-save` + `du -sh` for quantized sizes.
   - **Training**: example JSON under `models/common/training/_example/`, un-ignore in `.gitignore`, fast unit tests for training-adapter preview defaults.
   - Re-run `make lint`, `make test-fast`, then slow golden tests before merge.
7. **Finalize**
   - Re-run tests and basic perf checks after polish.
   - Add CLI/pipeline defaults and completions later, once core output is stable.
   - Ensure the model is wired into the standard surfaces:
     - `ModelConfig` entry + aliases
     - Thin model CLI entrypoint that uses shared parser/config/callback patterns
     - README following the structure and tone of existing model READMEs
     - Python API example that matches the CLI/defaults
   - Document any new mapping rules, shape constraints, or tolerances.

## Package layout (reference: `flux2`)

Use `src/mflux/models/flux2/` as the canonical tree. Do **not** invent flat mlx-vlm-style roots (`config.py`, `scheduler.py`, `fp8.py`, `layout.py`, monolithic `model/transformer.py`). Aliases and defaults live in `ModelConfig`; checkpoint validation belongs in the initializer and/or `*WeightDefinition`, not a separate layout module.

```
{model}/
  {model}_initializer.py
  __init__.py                    # export variant + initializer
  README.md
  cli/
    {model}_generate.py          # (+ edit/turbo CLIs when applicable)
  latent_creator/
    {model}_latent_creator.py
  model/
    {model}_text_encoder/        # prompt_encoder.py, tokenizer pieces, text_encoder.py
    {model}_transformer/         # attention, blocks, rope, transformer.py (split files)
    {model}_vae/                 # or reuse shared VAE (e.g. flux2_vae) — document in README
    {model}_scheduler/           # only when not covered by models/common/schedulers
  variants/
    __init__.py                  # re-export public variant class(es)
    txt2img/
      __init__.py
      {model}.py                 # e.g. flux2_klein.py, ideogram4.py
    edit/                        # when the model supports image-conditioned generation
      __init__.py
      {model}_edit.py
  weights/
    __init__.py
    {model}_weight_definition.py # components, download patterns, tokenizers
    {model}_weight_mapping.py    # WeightTarget list / key transforms for base weights
    {model}_lora_mapping.py      # LoRA key aliases (diffusers, PEFT, kohya) — when LoRA is supported
  training_adapter/              # when mflux-train is supported
    {model}_training_adapter.py
```

**Variants:** always place txt2img classes under `variants/txt2img/` (even for single-mode models). Use `variants/edit/` for edit/img2img variants. Import from the full path in `save.py` and CLIs, e.g. `variants.txt2img.flux2_klein`.

**Weights:** `*WeightDefinition` is required for every port. Add `*WeightMapping` when diffusers/HF key names need explicit targets. Add `*LoRAMapping` when inference or training supports LoRA — wire `--lora-paths` / `--lora-scales` through the shared parser and add fast tests that community LoRA filenames map to non-zero keys (see integration checklist below).

**Variant class style (post-refactor):** match `flux2_klein` / `z_image` — `prompt_cache`, `_predict` / `_decode_latents`, thin `generate_image`, prompt encoding in `{model}_text_encoder/prompt_encoder.py`.

**Skip when not applicable:** `variants/edit/`, `training_adapter/`, `{model}_lora_mapping.py`, local VAE package (if reusing another model family’s VAE). Document omitted features in the model README.

## Integration surfaces checklist (don’t forget)

Past [closed PRs](https://github.com/filipstrand/mflux/pulls?q=is%3Apr+is%3Aclosed) show the same wiring gaps recurring on every new model. Use this as a **tick list** alongside the workflow above — not every row applies to every model (e.g. skip vision-encoder rows for txt2img-only), but scan it before opening a port PR.

### Repo wiring (required for every new model family)

| Surface | What to do |
|---|---|
| `pyproject.toml` | Register `mflux-generate-<model>` (and edit/turbo variants if separate) |
| `ModelConfig` | Entry in `AVAILABLE_MODELS`: aliases, HF repo id, `num_train_steps`, guidance support, sigma shift, `transformer_overrides`, distilled vs base step defaults |
| `cli/defaults/defaults.py` | `MODEL_CHOICES` + `MODEL_INFERENCE_STEPS` |
| `models/common/cli/save.py` | Route `mflux-save` to the **correct variant class** (txt2img vs edit vs turbo — wrong class silently drops weights; see [#405](https://github.com/filipstrand/mflux/pull/405)) |
| Main `README.md` | Model table row + attribution line |
| `src/mflux/models/<model>/README.md` | Examples aligned with Flux2-style layout; disk sizes measured (`du`, `mflux-save`) |
| `src/mflux/assets/` | Hero/showcase image if other models have one (`git add -f` when `*.jpg` is gitignored) |

### Weights, tokenizer, download scope

| Surface | What to do |
|---|---|
| `*WeightDefinition` | `get_components()`, `get_download_patterns()`, `get_tokenizers()` — **only** list artifacts mflux actually loads |
| Weight mapping | Explicit `WeightTarget` list; verify tensor names/shapes against HF cache blobs |
| Tokenizer | Exercise local-path load, partial HF cache, and any special formats (protobuf, sentencepiece, etc.) — see [#383](https://github.com/filipstrand/mflux/pull/383), [#389](https://github.com/filipstrand/mflux/pull/389), [#390](https://github.com/filipstrand/mflux/pull/390) |
| Optional reference components | Document in README what the upstream pipeline includes but mflux omits (extra encoders, preprocessors, etc.) |

### LoRA

| Surface | What to do |
|---|---|
| `*LoRAMapping` | Support **multiple export key conventions** (diffusers, PEFT `.default.weight`, kohya/`diffusion_model.*` aliases) — silent zero-key loads were fixed repeatedly ([#376](https://github.com/filipstrand/mflux/pull/376), [#374](https://github.com/filipstrand/mflux/pull/374), [#397](https://github.com/filipstrand/mflux/pull/397)) |
| Tests | Fast tests that real community LoRA filenames map to non-zero keys |
| Inference CLI | `--lora-paths` / `--lora-scales` via shared parser (no bespoke loader) |

### Inference CLI & shared features

| Surface | What to do |
|---|---|
| Thin CLI | `CommandLineParser` + `CallbackManager.register_callbacks(...)` |
| `DimensionResolver` | Use in generate/edit CLIs when width/height can be omitted (API/OpenWebUI paths) — [#378](https://github.com/filipstrand/mflux/pull/378) |
| `latent_creator` | `pack_latents` / `unpack_latents`; img2img path must match txt2img normalization (BN, scale factor) |
| `tiling_config` | If initializer sets custom tiling, ensure `MemorySaver` does not overwrite it |
| Guidance defaults | Distilled vs base: match `ModelConfig`, CLI default, README, and training preview adapter |
| `mflux-save` round-trip | Save quantized model → load from local path → generate; confirm `model.safetensors.index.json` if sharded |

### Training (if supported)

| Surface | What to do |
|---|---|
| `training/runner.py` | Register `*TrainingAdapter`; handle `low_ram` / tiling like other models |
| Example JSON | `models/common/training/_example/train_<model>.json` + `.gitignore` un-ignore |
| Preview generation | Adapter should use canonical steps/guidance for distilled vs base (unit test this) |
| Local `model_path` | Confirm `mflux-train` works with saved local weights — [#370](https://github.com/filipstrand/mflux/pull/370) |

### Tests & CI

| Surface | What to do |
|---|---|
| Slow golden test | `tests/image_generation/test_generate_image_<model>.py` + `reference_*.png` on CI hardware |
| Fast tests | LoRA mapping, training-adapter preview defaults, argparser if new CLI flags |
| `make lint` / `make test-fast` | Before slow tests |

When fixing a gap for model *N*, ask whether the same gap exists for other recent models and whether a **shared** fix belongs in `models/common/` (preferred over copy-paste per model).

## Tooling expectations
- Use `uv` for running scripts and tests: `uv run <command>`.
- Prefer `uv run python -m <module>` for local modules.

## Deliverables
- Deterministic MLX test that verifies correctness.
- Documented weight mapping, shape constraints, and any known tolerances.
- Cleaned, shared components aligned with mflux style.
- Standard mflux surfaces in place: config aliases, thin CLI, and a README/examples pass aligned with the final behavior.
