---
name: mflux-testing
description: Run tests in mflux (fast/slow/full), preserve image outputs, and handle golden image diffs safely.
---
# mflux testing

This repo uses pytest with image-producing tests. Always preserve outputs for inspection and **never** update reference images unless explicitly asked.

## When to Use

- You need to run tests (fast/slow/full) or debug failing tests.
- There are image/golden mismatches and you need to report paths/output for review.

## Instructions

- Prefer the Makefile test targets:
  - `make test-fast` (fast tests, no image generation)
  - `make test-slow` (slow tests, image generation)
  - `make test` (default selection, skips slow model tests)
  - `make test-all` (full suite, slow tests download model weights)
- Always keep `MFLUX_PRESERVE_TEST_OUTPUT=1` on test runs (already built into the Makefile test targets).
- If a change affects defaults, config resolution, metadata fields, or CLI behavior, add or update tests that cover the changed behavior directly instead of relying only on manual verification.
- If tests fail:
  - Summarize the failing test names and the key assertion output.
  - Point to any generated images/artifacts on disk for manual review.
- Do **not** regenerate/replace reference (“golden”) images unless the user explicitly requests it.

## Updating golden images (new model or hardware refresh)

Golden tests compare generated PNGs to `tests/resources/reference_*.png` (typically 15% mismatch threshold).

**When to update** (only with explicit user approval):
- After validating the port on target hardware (CI Mac) via slow tests with `MFLUX_PRESERVE_TEST_OUTPUT=1`
- After choosing a stable prompt/seed/settings via diffusers comparison and/or latent-injection confidence (`mflux-debugging`)
- When old references used a bad prompt (e.g. ambiguous subject) or wrong seed for mflux’s RNG

**Workflow**:
1. Run slow test → inspect `tests/resources/output_*.png` vs `reference_*.png`
2. If output is correct but reference is stale, re-run generation with same test parameters and replace reference PNGs
3. Commit test + reference images together with a clear message (e.g. `test(<model>): update golden images for local hardware`)

**Important**: Golden tests lock **mflux-native** sampling (`mx.random` + mflux schedulers), not diffusers pixel parity. A good diffusers side-by-side or injected-latent run builds confidence in the model code; the golden still reflects mflux’s full recipe on CI hardware.

## Manual validation (config resolution + local model paths)

Use when a change touches model config resolution, `mflux-save`, or the model’s generate CLI, or when a PR fixes local model-path handling for the model under investigation. Refer to the `mflux-cli` skill to find the correct generate command for the model you are testing.

- Run a local-path quantize/save:
  - Use the `mflux-cli` skill to look up the correct command and flags.
  - Verify CLI usage with the command’s `--help` before running it.
  - Save to a known location (e.g., Desktop) to make follow-up steps explicit.
- Run generation from the saved model using the correct model-specific generate CLI:
  - Use the `mflux-cli` skill to find the generate command and required flags.
  - Verify CLI usage with the command’s `--help` before running it.
- If the model has multiple size variants, repeat the above for each variant to confirm the correct overrides are applied.
- Do not commit output artifacts; delete or leave them untracked.
