# Model compatibility check

## Purpose

`mflux-check model` verifies safetensors package structure. For a known MFlux
model layout, it also verifies the required parts for the supported MFlux CLI.
For an unrecognized safetensors layout, `pass` reports only a structural check.

The command supports both Hugging Face safetensors packages and MFlux-saved
packages. A package passes only when MFlux can identify a supported model
layout and all required model parts are available.

## Command interface

```text
mflux-check model <model-reference>
mflux-check model --fix <model-reference>
```

Examples:

```text
mflux-check model black-forest-labs/flux-3-image-klein
mflux-check model ./models/flux-3-image-klein
mflux-check model mflux-community/qwen-image-edit-2511-mflux-q6
mflux-check model --fix ./models/qwen-image-edit-2511-mflux-q6
```

There is no `--format` option. The command determines the format from the
resolved model package.

## Model reference resolution

Resolution follows the existing model path rules:

- An existing local path is a local model package.
- A value in `org/model` form is a Hugging Face repository reference.
- A value that starts with `./`, `../`, or `~/` is always a local path.

For a Hugging Face reference, the normal check reads only the default Hugging
Face cache. It does not download files, contact Hugging Face, or repair the
cache. A missing or incomplete cache is a check failure.

## Format and model identification

The checker reads safetensors headers and tensor key ordering. Hugging Face
safetensors and MFlux safetensors use different key ordering.

The checker must:

1. Find the model safetensors files and their shard index files.
2. Identify the package as MFlux or Hugging Face safetensors.
3. Match the observed layout to a model supported by the MFlux CLI.
4. Determine the components, subdirectories, and shards required for that
   model.

A safetensors package with no known MFlux CLI layout passes the structural
check. The output does not claim that an MFlux CLI can run that package.

## Validation

For both formats, the checker checks:

- Every required component is present.
- Every shard named by a safetensors index exists.
- Index files are readable and name only local shard files.
- Safetensors headers are readable.
- When MFlux recognizes the layout, each package layout matches its supported
  MFlux CLI model.

For MFlux packages, the checker also checks MFlux metadata such as
`mflux_version` and `quantization_level` where required by the format.

The checker counts only model weight files for `size`. It does not count
tokenizer files, configuration files, or duplicate Hugging Face cache blobs.
The displayed unit is decimal `GB`.

## Results

Successful output is concise:

```text
#################
MFlux Check Model:
 check: pass
 format: safetensors
 size: 13.35 GB
```

An incomplete MFlux package reports a warning and gives the repair command:

```text
#################
MFlux Check Model:
 check: warn
 format: mflux
 size: 9.21 GB
 model: flux-3-image-klein
 problem: Missing component: transformer.
 problem: Missing component: vae.
 fix: mflux-check model --fix ./models/flux-3-image-klein
```

A package fails when it cannot run with an MFlux CLI. The failure output must
name the failed check and the specific missing, invalid, or unsupported part.

## Repair mode

`mflux-check model --fix <model-reference>` is available when the checker can
identify an incomplete MFlux package and the exact parts that it requires.
After an attempted repair, it runs the full check again and reports either
`pass` or the remaining problems.

The source and download policy for missing files is not decided yet:

- The normal check is strictly offline and never downloads.
- Repair restores a missing cache shard only when the snapshot entry is a link
  to a missing blob in the same cache repository. The blob name is the SHA-256
  of the file. Repair copies a shard from another local snapshot only when its
  SHA-256 is equal to the blob name.
- Repair does not write through a link that points outside the blobs directory
  of the cache repository.
- Repair does not restore a shard that has no expected hash. Matching tensor
  names do not prove that two revisions contain the same weights.
- Repair of genuinely absent files needs an explicit source policy before
  implementation. It must not silently download files.