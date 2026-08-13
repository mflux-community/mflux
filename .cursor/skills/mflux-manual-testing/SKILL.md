---
name: mflux-manual-testing
description: Manually validate mflux CLIs by exercising the changed paths and reviewing output images/artifacts.
---
# mflux manual testing

Some regressions (especially in CLIs and image IO) are easiest to catch by running the commands and visually inspecting outputs. This skill provides a lightweight, change-driven manual test checklist.

## When to Use

- You changed any CLI entrypoint(s) under `src/mflux/models/**/cli/`.
- You touched callbacks (e.g. stepwise output, memory saver) or metadata/image saving.
- Tests are green but you want confidence in real command usage.

## Strategy (change-driven)

- Identify what changed on your branch (new flags, default behavior changes, new callbacks, new models).
- Only run manual checks for the touched areas; don’t try to exercise every CLI.
- Prefer 1–2 seeds and a small step count (e.g. 4) for fast iteration, unless the change affects convergence/quality.
- Before manual CLI testing, **reinstall the local tool executables** so you’re testing the latest code:

```bash
uv tool install --force --editable --reinstall .
```

## Core CLI checks (pick what’s relevant)

- **Basic generation**: run the CLI once with a representative prompt and confirm the output is not “all noise”.
- **Model saving** (if relevant): if you touched weight loading/saving or model definitions, run `mflux-save` for the affected model(s) and verify:
  - the output directory is created
  - the command completes without missing-file errors
- **Run from disk** (if relevant): if you touched save/load paths or model resolution, generate from a locally saved model directory by passing `--model /full/path/to/saved-model` and confirm it runs and produces a sane image.
- **Stepwise outputs** (if relevant): run with `--stepwise-image-output-dir` and confirm:
  - step images are written for each step
  - the final step image matches the final output image qualitatively
  - the composite image is created
- **Low-RAM path** (if relevant): run with `--low-ram` and confirm:
  - generation completes
  - output quality is sane (no unexpected all-noise output)
- **Metadata** (if relevant): run with `--metadata` and confirm the `.metadata.json` sidecar is emitted and looks consistent.

## Output review (human-in-the-loop)

- Always point the human reviewer at:
  - the final output image path
  - any stepwise directory / composites
  - any metadata JSON files
- Ask the human to visually confirm “looks correct” rather than attempting pixel-perfect parity manually.

## diffusers reference comparison (new model ports)

mflux does not install diffusers; use a sibling clone (commonly `../diffusers` on Desktop).

**When**: validating a new MLX port before merge, or when golden tests / visuals look wrong.

**Match settings**: same prompt, width, height, seed, steps, guidance. For fair speed comparisons, match precision (mflux bf16 vs diffusers bf16, not `-q 8` vs bf16).

**diffusers setup tips** (read the reference pipeline first):
- Compare `model_index.json` / `from_pretrained` kwargs with mflux `get_download_patterns()` — disable or pass `None` for components mflux does not load
- `local_files_only=True` / `HF_HUB_OFFLINE=1` to use cache and surface missing files early
- Distilled vs base variants are often the same pipeline class with different checkpoint + steps/guidance

**Run both**:
```sh
# mflux
uv run mflux-generate-<model> --prompt "..." --width 640 --height 368 --seed 7 --steps 8 --guidance 1.0 --output /tmp/mflux.png

# diffusers (from diffusers repo)
cd ../diffusers && HF_HUB_OFFLINE=1 uv run python -c "..."  # inline script; see mflux-debugging
```

**If visuals differ**: do not assume the port is broken. Run the **latent injection** workflow in `mflux-debugging` to separate (a) transformer/VAE quality from (b) RNG/scheduler differences.

Save comparison PNGs to explicit paths; report `/usr/bin/time -p` totals. Do not commit comparison artifacts.

## Notes

- If the installed `uv tool` executable behaves differently from `uv run python -m ...`, prefer the local module run to isolate environment/tooling issues.
- If you need to reinstall the local tool executables, see the repo rules for the current recommended command.

