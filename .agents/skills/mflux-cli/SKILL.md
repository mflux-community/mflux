---
name: mflux-cli
description: Navigate MFLUX CLI capabilities, locate commands by area, and summarize supported features.
---
# mflux CLI navigation

Use this skill to inventory CLI capabilities, summarize what the CLI supports, and guide where to look
for commands without relying on brittle file paths.
Because README examples can drift, prefer verifying support against the current CLI entrypoints.

## When to use

- You need to list supported CLI features or commands.
- You need to find where a capability is implemented in the CLI.
- You are documenting or refactoring CLI features and want a stable map.
- A user asks for CLI help, e.g., “Can you help me generate an image using z-image?”, “Which model is best?”, etc.

## How to find commands (structure, not exact paths)

- Common/shared CLI arguments live in the central CLI parser module.
- Model-specific CLI entrypoints live under each model's CLI package.
- Repo-level CLI helpers (completions, defaults) live under the shared CLI package.
- Utilities may add standalone CLIs (e.g., metadata info, LoRA library).

## Best practices when constructing CLI calls

- **Inference steps**: When constructing a CLI call for any model, *always* check `MODEL_INFERENCE_STEPS` and use the model's default/recommended step count unless the user explicitly asks for a different value. This is especially important for distilled/non-base variants (for example `fibo-lite`, `z-image-turbo`, and other `*-turbo`/`*-lite` models), where using full/base-model step counts is usually counterproductive.
- **Resource/inspection flags**: Mention `--low-ram` to reduce memory usage and `--stepwise-image-output-dir` for stepwise outputs when useful.
- **Python API requests**: If a user asks for the Python API, treat the equivalent CLI script as the best starting reference for the underlying parameters and defaults.
- **CLI implementation changes**: When adding or changing CLI behavior, prefer extending shared parser methods and shared helpers before adding manual one-off arguments or save paths in a model CLI.
- **Docs/examples drift**: If CLI defaults, supported flags, or recommended usage changed, update the matching README examples in the same pass.

## Capability inventory (current)

### Core generation

- Text-to-image across Flux, Flux2, Qwen, Z-Image Turbo, FIBO.
- Image-to-image where supported (Flux, Qwen, Z-Image Turbo, FIBO).

### Editing and conditioning

- Kontext image conditioning.
- In-context editing and reference-image workflows.
- CATVTON (virtual try-on).
- Redux multi-image conditioning.
- ControlNet (Canny).
- Depth conditioning.
- Fill / inpainting.
- Flux2 Edit and Qwen Edit (multi-image edit).

### Upscaling

- SeedVR2 diffusion upscaler (preferred).
- Flux ControlNet upscaler (legacy).

### Model management

- Quantized inference and saving quantized models.
- Local model path loading (with base-model hints when needed).

### LoRA

- Load LoRAs, multi-LoRA, scale control.
- In-context style LoRA shortcuts.
- LoRA library lookup tool.

### Metadata and reproducibility

- Export JSON metadata per image.
- Reuse prior parameters from metadata config files.
- Inspect metadata from existing images.

### Prompt tooling

- Prompt files.
- Negative prompts where supported (not supported for Flux2).

### Training

- LoRA finetuning.

### Utilities

- DepthPro depth-map extraction.
- FIBO VLM prompt inspire/refine tools.
