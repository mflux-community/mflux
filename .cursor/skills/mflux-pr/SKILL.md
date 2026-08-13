---
name: mflux-pr
description: Make a clean PR in mflux (inspect diff, quick verification, commit, push, open PR) using repo conventions.
---
# mflux pull request workflow

## When to Use

- You’re about to open a PR (or want a safe sequence to do it).

## Instructions

- If you run tests as part of PR hygiene, prefer fast tests first:
  - `make test-fast`
- Keep commits focused and messages consistent with repo history.
- If the PR changes CLI defaults, public APIs, or model behavior, check for README/example drift before opening the PR.
- **Always ask for permission** before pushing to the remote repository.
- If `gh` isn’t available, fall back to the GitHub web UI (or stop and ask).

## Pre-merge checklist (model port PRs)

Use after the core port lands and you are polishing for merge. For the full **integration surfaces** tick list (LoRA key formats, save routing, tokenizer edge cases, etc. learned from past closed PRs), see `mflux-model-porting` → *Integration surfaces checklist*.

### Correctness

1. `make lint` and `make test-fast`
2. Slow golden tests for the new model:
   ```sh
   MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest tests/image_generation/test_generate_image_<model>.py -m slow -v
   ```
3. Optional but high-signal: diffusers side-by-side + latent injection (`mflux-debugging`, `mflux-manual-testing`)

### Cross-model diff audit

List files changed outside `src/mflux/models/<model>/`:

| Category | Expected |
|---|---|
| `pyproject.toml`, `cli/defaults/defaults.py`, `ModelConfig`, `mflux-save` routing | Required wiring |
| `README.md` table + attribution | Required |
| Training `runner.py`, example JSON, `.gitignore` JSON exceptions | If training supported |
| Shared VAE/callback/training one-liners | Only if required; document blast radius in PR |
| Personal `.gitignore`, unrelated formatting | **Remove** |

Verify quantized README disk claims with measurement:
```sh
du -sh ~/.cache/huggingface/hub/models--<org>--<Model>*
mflux-save --model <alias> --quantize 8 --path /tmp/model-q8 && du -sh /tmp/model-q8
```

### Docs / examples

- Model README matches a recent port (e.g. Flux2): hero image, turbo + base CLI, feature section, disk warning, Notes, Training.
- Main `README.md` model table row (correct release date).
- Showcase asset if other models have one (`src/mflux/assets/`; may need `git add -f` when `*.jpg` is gitignored).

### PR description callouts

- Shared code touched and why (shared VAE, callbacks, training runner, etc.).
- Reference pipeline features intentionally **not** ported (optional preprocessors, extra encoders, components omitted from mflux weight downloads).
- Known non-parity with diffusers (RNG, sigma schedule, optional modules) if golden tests lock mflux-native sampling.

