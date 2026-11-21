# PRD — Stage 3d finish: confirm on ANE, then images

**Branch:** `experiment/zit-on-ane-stage3d` (from `7c43f8c7`)
**Report:** append to `stage3d/3D-ROWSPLIT-MLP.md`
**Stop point:** after Step 3. Hand over the images and wait.

## Decisions (from Scriv)

- **Accuracy bar:** per-row int8 at ~0.99998 block cosine is accepted.
- **Dropped:** the q8 block-cosine gate, g64h and smoothing. Don't pursue them.
- Keep it simple. Don't chase the last decimal.

## Step 1 — Range fix on the ANE

- Put the a+b range fix (1/√K dequant constant + per-row `h` scaling, restored in the fp32 GPU epilogue) into the real program and bridge, with per-row scales.
- Run one T1 pass over all 34 real blocks, at share 0.50 and g ∈ {1, 2, 4}.
- **Pass:** block cosine ≈ the mirror value (~0.99998), and peak |partial sum| ≤ 0.5 × 65504 (check magnitude, not finiteness).
- Pick the best share/g for Step 2.

## Step 2 — Wire into generation + re-time

- **Reboot first** if swap is above ~0.
- Wire the fused row-split MLP into the real Z-Image Turbo generation path behind a spike flag (e.g. `--ane-mlp`).
- Re-run T4 (chained 34-block step): GPU-only bf16 vs ANE.
- Also time one full image (1024², 8 steps) for each of **bf16, q8 and ANE**.
- Record: step time, image time, swap delta (must be 0), and wired delta. The memory fix comes later; just report the numbers.

## Step 3 — Images (Scriv has approved this run)

- 4 prompts × 1 fixed seed, 1024², 8 steps.
- Three arms: **bf16, q8, ANE** (the Step 2 config).
- Save as `stage3d/images/<prompt#>_<arm>.png`, plus one side-by-side grid per prompt (`<prompt#>_grid.png`, with arm labels).
- Add a small table: image time per arm, PSNR and LPIPS of q8 and ANE against bf16.
- Prompts: one portrait, one text-on-sign, one detailed landscape, one busy indoor scene. Write them into the report.

## Then stop

Report the paths and the table, then wait for Scriv's eyeball verdict.
Not in this PRD: the single-weight-copy memory fix (next).

## Rules (unchanged)

- Foreground or Interactive QoS only; hold `s3a.LOCK` (don't double-acquire inside the wrapper); `watchdog.sh` armed; `macmon` ≈ 9 W.
- Any swap growth during a timed run invalidates it.
- Take every number and hash from tool output.
- Commit locally. Ask before any `git push` ("CodeRabbit locally first?"). Fork only.
