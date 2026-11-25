# Task: Write the benefit summary for a pull request

Write a short summary in Simplified Technical English. The summary tells the readers how a pull request (PR) benefits the end users of MFlux, improves the application, or benefits the project.

Input: the current branch. Optional: PR number `{PR_NUMBER}`, if the PR is open.

## 1. Collect the facts

Use only facts that you can find in the branch or the PR. Do not guess.

1. Read the commit messages. Use `git log --oneline main..HEAD`. If the PR is open, also read the PR title and the PR description.
2. Read the full diff. Use `git diff main...HEAD`. If the PR is open, you can use `gh pr diff {PR_NUMBER}`.
3. Read the list of changed files. Use `git diff --stat main...HEAD`. Sort each file into one of these groups:
   - Source code
   - Tests
   - Documentation
   - CI/CD configuration
   - Build or packaging
4. If the PR is open, read the CI status. Use `gh pr checks {PR_NUMBER}`.
5. Read the linked issues, if the PR has them.
6. For each change in source code, find what the user sees. Examples: a new flag, a changed default, a fixed error, a faster run, a better image.

## 2. Find each benefit

Examine each type of benefit in this list. Write a benefit only if the diff shows it.

| Benefit type | Evidence that you must find |
| --- | --- |
| Bug fix | Code that changes wrong behavior. A linked bug issue. A test that failed before the change. |
| New feature | A new flag, a new command, a new function, or a new model. |
| CLI change | A new, removed, or renamed argument. A changed default value. A changed output. |
| Python API | A new or changed public class, method, or function. |
| Performance | Less time, less memory, or fewer downloads. Use a number only if the PR gives one. |
| Output quality | A change to the images, text, or files that the tool makes. |
| Reliability | Better error messages. Better input checks. Fewer crashes. |
| Tests | New tests, or tests that now cover more cases. |
| Documentation | New or fixed README text, docstrings, or examples. |
| CI/CD | New or changed workflow steps, checks, or release automation. |
| Maintenance | Code that is easier to read, change, or reuse. |
| New model support | Code that adds support for a new model. |

## 3. Find the beneficiaries

Name each group of people that gets a benefit. Use these groups:

- 'CLI users'
- 'Python developers' who import the package
- 'Downstream developers' of UIs or other apps that use the package
- 'Contributors' and maintainers of the project
- 'Release managers'

Some changes have no effect that a user can see. Examples: a refactor, a cleanup, a tool upgrade. For these changes, the beneficiary is 'Contributors' and the benefit type is Maintenance.

## 4. Select the primary benefit

1. Rank the benefits by how much they change the experience of the largest affected group.
2. A bug fix or a change that users can see ranks above tests, documentation, or CI.
3. The primary benefit is the benefit at the top of the list.
4. The primary beneficiary is the group that gets the primary benefit.

## 5. Write the summary

Use this structure:

1. **Headline.** One sentence. Name the primary beneficiary and the primary benefit.
2. **Primary benefit.** Two or three sentences. Tell what the reader can now do, or what now works.
3. **Other benefits.** A bulleted list. Start each bullet with the beneficiary group in bold. One benefit for each bullet.
4. **Action for the reader.** One sentence, only if the reader must do something. Examples: a renamed flag, a new dependency, a changed default.

## 6. Obey these rules

- Write only about what the PR adds or changes.
- Do not write about work that the PR does not do.
- Do not write about features that the PR keeps the same.
- Do not write about future work.
- Do not write "no change" statements, except in one case. If the PR changes nothing that CLI users can see, write one sentence that says so. Put it in the "Other benefits" list.
- Use the words that a user knows. Explain a technical term the first time you use it.
- Use the same word for the same thing every time.
- Do not use marketing words. Examples: `seamless`, `powerful`, `robust`, `easy`, `simple`, `great`.
- Use a number only if the PR, its tests, or its CI give that number.

## 7. Examine your summary

Before you send the summary, do these checks:

1. Make sure that the headline names a beneficiary and a benefit.
2. Make sure that each claim has evidence in the diff, the PR description, or CI.
3. Remove each sentence about work that the PR does not do.
4. Apply the `asd-ste100` skill in STE-flavored mode. Run `python3 .cursor/skills/asd-ste100/scripts/ste-lint.py --disable synonym-rotation` on the summary. Fix each hard violation.

## Example

Input: a PR that adds `validate`, `load`, and `generate` methods to the Qwen-Image commands.

Output:

> **Python developers and downstream developers can now load a Qwen-Image model one time and make many images from it.**
>
> The `mflux-generate-qwen` and `mflux-generate-qwen-edit` commands now have three Python methods. `validate()` returns the model config. `load()` builds the model. `generate()` makes one image and does not save it. An app does not load the model again for each image.
>
> - **Python developers:** `generate()` now uses the same guidance default as the CLI.
> - **Python developers:** The Qwen README shows the new methods. It also tells how much memory a loaded model uses.
> - **Contributors:** 41 new fast tests lock the CLI behavior. CI now finds changes that break these commands.
> - **CLI users:** The flags, defaults, and output do not change.
