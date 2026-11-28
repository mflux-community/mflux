# A/B benchmark

`ab_benchmark.py` compares two or more ways to run the same mflux command. Use it to measure the effect of a flag, a quantization level, or a code change on one Mac.

The script runs each arm in a new process. It makes one image per process. It changes the order of the arms between runs. This method removes the effect of a slow start or a hot Mac on one arm.

## Requirements

- An Apple Silicon Mac with mflux installed.
- The model weights on disk. The script does not download them.
- Free memory for the model. Close other large applications first.

## Quick start

Run the command from the repository root:

```bash
uv run python scripts/benchmarks/ab_benchmark.py \
    --arm "stock=mflux-generate-z-image-turbo --model z-image-turbo -q 8 --steps 8 --seed 1 --output {image}" \
    --arm "lowram=mflux-generate-z-image-turbo --model z-image-turbo -q 8 --steps 8 --seed 1 --low-ram --output {image}" \
    --case "fox=--prompt 'a red fox in the snow'" \
    --reps 1 \
    --out-dir /tmp/ab
```

The script prints one JSON line for each run. At the end it prints a report in Markdown.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--arm NAME=COMMAND` | required | One command line. It must contain `{image}`. Repeat for each arm. |
| `--case NAME=EXTRA_ARGS` | one empty case | Arguments added to the end of every arm. Repeat for each case. |
| `--reps N` | `3` | Number of times to run each arm and case. |
| `--out-dir DIR` | required | Folder for the images, `rows.json`, and `report.md`. |
| `--min-free-percent N` | `3` | Stop a run when free memory is below N percent. |
| `--swap-limit-mb N` | `6000` | Stop a run when swap grows by more than N MB. |
| `--discard-swap-mb N` | `64` | Keep a run in the table but leave it out of the medians when swap grows by more than N MB. |
| `--lock-file PATH` | `~/.cache/mflux/benchmark.lock` | Lock file that stops two benchmarks from running at once. |
| `--no-lock` | off | Do not use the lock file. |
| `--psnr` | off | Compare the rep 0 image of each arm with the first arm. Needs `numpy` and `Pillow`. |

## How to write an arm

- The script replaces `{image}` with the output path. Every arm must pass it to the command, for example with `--output {image}`.
- The first arm is the baseline.
- A case adds its text to the end of every arm. Use cases for prompts or other inputs that do not change the arm.
- Arms and cases are names. Use names without spaces.

## Read the output

Each run has these values:

| Column | Meaning |
|---|---|
| `wall s` | Time for the whole process. It includes the model load. |
| `gen s` | `generation_time_seconds` from the metadata file. It is empty when the command writes no metadata file. |
| `first step s` | Time for the first denoising step. It includes one-time costs, such as compile and weight loads. |
| `steady step s` | Median time of the other steps. |
| `swap grew MB` | Growth of swap during the run. |

The report shows the median of each arm. It also shows a ratio for each arm against the baseline. A ratio above 1 means that the arm is faster than the baseline.

Step times come from the progress bar that mflux prints on stderr. A command with no progress bar gives no step times.

## Safety

The script stops a run when memory use is too high. It kills the whole process group of the run.

- Free memory below `--min-free-percent`: the script kills the run.
- Swap growth above `--swap-limit-mb`: the script kills the run.
- Swap growth above `--discard-swap-mb`: the table keeps the run, and the medians leave it out.

A killed run gets `ok: false`. The script stops with exit code 1 when any run fails.

The lock file stops a second benchmark from starting while one runs. If the holder process is no longer alive, the script takes over the lock. Otherwise it waits up to one hour.

## Tests

The unit tests use a fake command. They load no model.

```bash
just test-fast
```

To run only the benchmark tests:

```bash
uv run --no-sync python -m pytest tests/benchmarks/test_ab_benchmark.py -m fast -o addopts=""
```

The `-o addopts=""` option avoids the default pytest options in `pyproject.toml`. Those options select `--failed-first`, which the no-sync environment does not provide.

## Output files

The script writes these files to `--out-dir`:

- `<arm>_<case>_r<rep>.png`: the image of each run.
- `<arm>_<case>_r<rep>.metadata.json`: the metadata sidecar, when the command writes one.
- `rows.json`: one object for each run.
- `report.md`: the report.

Keep the output folder outside the repository. The images are large.
