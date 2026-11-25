# Task: Write the release note for a pull request

Write the release note for one pull request (PR). The release note goes into the GitHub release for the next MFlux version.

Input: the `What` section that you wrote, and the facts that you collected for it.

## 1. Know the reader

The reader is a user of MFlux. The user reads the GitHub release. The user does not read the PR.

## 2. Know how the release harvester uses the note

`src/mflux/release/release_notes.py` reads the fenced `release-note` block of each merged PR. It does these steps:

- It joins all lines of the note into one line.
- It adds the PR number at the end, for example `(#780)`.
- It puts the note in a section. The section comes from the first PR label that matches:

| PR label | Section |
| --- | --- |
| `breaking change` | Breaking |
| `bug` | Fixed |
| `feature suggestion`, `enhancement` | Added |
| `improvement` | Improved |
| `documentation` | Docs |
| `chore`, `ci` | Internal |
| No matching label | Changed |

## 3. Write a note for each PR

Each PR gets a note. Do not write `none`.

1. If the PR changes something that CLI users, Python developers or downstream developers can see, write the note for them.
2. If the PR changes only what contributors see, write the note for contributors. The note goes in the Internal section. Examples: a new agent skill, a new CI check, a new `just` recipe, a test-only change, a refactor.

## 4. Select the label

Tell the developer which label the PR needs. Use the table in step 2. Do not add the label yourself.

## 5. Write the note

1. Write one or two sentences. Approximate maximum of 180 characters.
2. Start with the thing that changed. Name the command, the flag, the model or the Python method in backticks.
3. Tell what the user can now do, or what now works.
4. For a fix, tell what went wrong before the fix.
5. For a renamed or removed flag, tell what the user must do. Tell if the old name still works.
6. Do not write the PR number. The harvester adds it.
7. Do not write "This PR".
8. Do not use marketing words. Examples: `seamless`, `powerful`, `robust`, `easy`, `simple`, `great`.
9. Use a number only if the PR, its tests or its CI give that number.

## 6. Write the note two times

Put the same note in two places, below the `## Release note` heading:

1. As plain text, above the opening fence. Reviewers read this copy.
2. Inside the fenced `release-note` block. The harvester reads this copy.

If you change the note, change both copies.

## 7. Examine the note

1. Make sure that a user who did not read the PR understands the note.
2. Make sure that each claim has evidence in the diff, the PR description or CI.
3. Apply the `asd-ste100` skill in STE-flavored mode. Run `python3 .cursor/skills/asd-ste100/scripts/ste-lint.py --disable synonym-rotation` on the note. Fix each hard violation.

## Examples

These notes come from the MFlux 0.21.0 release.

Added (#780):

> `mflux-generate-z-image-turbo` now exposes `ZImageTurboCommand.load` and `ZImageTurboCommand.generate`, so Python code can load the model once and generate many images with the command's own flag handling.

Fixed (#758):

> mflux now refuses a checkpoint whose weight names do not match the model, with an error that names the part and the missing weights. Before, such a checkpoint loaded with random weights and produced noise. This mostly affects checkpoints converted for other programs.

Changed (#797):

> Three CLI flags have new names: `--make-conf` (was `--metadata`), `--no-exif` (was `--no-metadata`) and `--config-from-conf` (was `--config-from-metadata`). The old names continue to work as aliases.
