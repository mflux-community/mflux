---
name: mflux-pr
description: Make a clean PR in mflux (inspect diff, quick verification, commit, push, open PR) using repo conventions.
---
# mflux pull request workflow

## When to Use

- You’re about to open a PR (or want a safe sequence to do it).

## Instructions

- If you run tests as part of PR hygiene, prefer fast tests first:
  - `just test-fast`
- Keep commits focused and messages consistent with repo history.
- If the PR changes CLI defaults, public APIs, or model behavior, check for README/example drift before opening the PR.
- **Always ask for permission** before pushing to the remote repository.
- If `gh` isn’t available, fall back to the GitHub web UI (or stop and ask).

## PR body and release note (required)

The `release-note` CI check fails the PR if the body has no complete release-note block. Do these steps for every PR:

1. Start the PR body from `.github/pull_request_template.md`. Do not write the body from nothing.
2. Fill in the `release-note` block. The opening fence must be exactly ` ```release-note `, on its own line. Put the note on the lines below the fence, not on the fence line.
3. Write one or two sentences that a user can read. For changes that users do not see (CI, tests, docs), write `none`.
4. Before you run `gh pr create`, run the same check as CI on the body file:
   ```sh
   python3 - body.md <<'PY'
   import re, sys
   body = open(sys.argv[1]).read()
   m = re.search(r"^```release-note[ \t\r]*\n(.*?)^```[ \t\r]*$", body, re.DOTALL | re.IGNORECASE | re.MULTILINE)
   sys.exit(0 if m and m.group(1).strip() else "FAIL: no complete release-note block")
   PY
   ```
5. If the check fails on an open PR, edit the PR body (`gh pr edit <n> --body-file body.md`). The check runs again on an edit. You do not need a new commit.

Keep the regex the same as `.github/workflows/release-note.yml` and `_FENCE` in `src/mflux/release/release_notes.py`.

## Pre-merge checklist (model port PRs)

Use after the core port lands and you are polishing for merge. For the full **integration surfaces** tick list (LoRA key formats, save routing, tokenizer edge cases, etc. learned from past closed PRs), see `mflux-model-porting` → *Integration surfaces checklist*.

### Correctness

1. `just lint` and `just test-fast`
2. `just ci-extract` after changing `AVAILABLE_MODELS` or `scripts/ci_extract_models.py`
3. Slow golden tests for the new model:
   ```sh
   MFLUX_PRESERVE_TEST_OUTPUT=1 uv run pytest tests/image_generation/test_generate_image_<model>.py -m slow -v
   ```
4. Optional but high-signal: diffusers side-by-side + latent injection (`mflux-debugging`, `mflux-manual-testing`)

### Cross-model diff audit

List files changed outside `src/mflux/models/<model>/`:

| Category | Expected |
|---|---|
| `pyproject.toml`, `cli/defaults/defaults.py`, `ModelConfig`, `mflux-save` routing | Required wiring |
| `scripts/ci_extract_models.py` `OVERLAY` / `EXTRA_ENTRIES` | Required CI-manifest wiring for every new model key or standalone tool |
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

