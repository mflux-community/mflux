# FLUX 3 memory measurements

Use `benchmark_memory.py` to compare the existing default and `--low-ram` CLI modes in fresh processes. The wrapper records phase timings and memory without changing the production loading policy.

## Reproduce

Use a saved local checkpoint and a unique report and image path for each run:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/time -l uv run python scripts/flux3/benchmark_memory.py \
  --report /tmp/flux3-lowram-1.json -- \
  --model /path/to/saved-flux3-klein-q4 --base-model flux3-image-klein \
  --low-ram --width 512 --height 512 --steps 16 --seed 42 \
  --prompt 'A small red ceramic robot standing beside a green fern on a wooden desk, soft morning window light, detailed handmade textures, product photograph.' \
  --output /tmp/flux3-lowram-1.png --make-conf
```

Run in this order: low-RAM, default, default, low-RAM. Remove `--low-ram` for default runs. Use a fresh process and unique filenames each time. Keep checkpoint, precision, prompt, dimensions, steps and seed fixed. The local checkpoint must include the tokenizer files. For a remote model, remove the offline variables and download it before measuring.

The current CLI releases the text encoder after encoding in both modes. For a single seed, default mode retains the transformer for decoding. Low-RAM mode releases it and sets a 1 GB allocator cache limit. Low-RAM also enables applicable VAE tiling defaults. This comparison measures those existing behaviours together.

## What the report measures

- `construct` includes model construction. MLX loads lazily, so later encoding and denoising also include weight reads and materialisation.
- `encode` and `decode` materialise their results. All phase boundaries synchronise Metal. This instrumentation changes timing; compare runs with the same wrapper.
- `before_loop` and `after_loop` measure the existing memory callbacks. The interval from `before_loop_end` to `after_loop_start` estimates denoising, including loop setup and callbacks.
- `mlx_lifetime_peak_bytes` retains the maximum peak across reset segments, including the peak before low-RAM resets the counter. It measures MLX allocation, not total system memory.
- `process_peak_rss_bytes` is the separate macOS lifetime RSS counter, in bytes. Do not add it to MLX allocation or treat it as a replacement for the Metal memory metric.
- `python_wall_s` excludes interpreter and uv startup. `/usr/bin/time -l` supplies external process wall time.
- System swap and memory pressure include every process. They do not isolate the benchmark. Filesystem cache is not flushed; repeated processes are not proven cold-cache runs.
- `versions` comes from installed distribution metadata. `source_path` identifies the imported generation CLI, and `source_sha` records its checkout revision (null for an installed package outside Git). `benchmark_source_sha` separately records the wrapper's checkout revision. The installed distribution can be older than the imported source selected with `PYTHONPATH`.

Reports include local paths and prompts. Review them before sharing. The wrapper restores its instrumentation on exit, writes a failure report for CLI errors, and refuses to replace an existing report. Success requires `saved_images` to equal `expected_images`, the number of seeds resolved by the CLI (including automatic seeds and metadata). An interrupted run fails even if earlier images were saved.

## Follow-up scope

Start with these baselines before proposing additional staged loading. Test BF16 and larger images separately, under recorded memory pressure. A loading change should preserve outputs for fixed inputs, reduce the relevant memory peak, and disclose extra I/O or latency. Repeated prompts and seeds, changed prompts, Base guidance and reference-image editing need their own validation before a general residency policy is proposed.

## M1 Max 32 GB results, 17 October 2026

Production source: `ec705857d20846f5636fdc4acaa7507fa9855782` (MFLUX 0.23.0), macOS 14.6, MLX 0.32.0. The existing Python environment had MFLUX 0.22.0 distribution metadata; `PYTHONPATH` selected the pinned source.

The checkpoint was a local q4 conversion from our earlier MFLUX port, stored on an external SSD. The [machine-readable report](benchmarks/m1-max-32gb-q4-20261017.json) includes file hashes, settings, phase timings and memory samples.

| Run order | External wall, s | Python wall, s | Denoise interval, s | Peak MLX, GB | Peak RSS, GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| lowram-1 | 120.29 | 119.09 | 105.99 | 4.102 | 1.811 |
| default-1 | 142.85 | 133.75 | 115.93 | 6.846 | 1.621 |
| default-2 | 114.83 | 111.46 | 91.73 | 6.846 | 2.612 |
| lowram-2 | 132.72 | 131.70 | 115.01 | 4.102 | 2.477 |

GB in the table means decimal gigabytes. All four 512×512 images have identical RGB pixels. The low-RAM runs use about 40% less peak MLX allocation than the default runs for this case.

Other applications were active. System-wide swap changed substantially across the run sequence; see the report for each value. These timings are observations under shared load, not a speed ranking. The q4 case demonstrates that the existing low-RAM path reduces allocation. It does not establish a need for another loading policy.

The ten recorder tests pass, including interrupted multi-seed runs and imports from a separate checkout or an installed package. Three initial counterexamples (losing a pre-reset peak, skipping lazy evaluation, and leaking instrumentation) each failed a relevant test. Restoring the two reporting bugs also fails the new regression tests; the restored recorder passes again.
