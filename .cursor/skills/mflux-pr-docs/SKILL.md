---
name: mflux-pr-docs
description: Write the PR body for the current mflux branch and show it to the developer for approval. Fills the What section, the release-note block and the Verification section of the PR template. Use when the developer asks to write, fill or update a PR description or a release note.
---

# mflux PR docs

Write the PR body for the current branch. The developer must approve the body before it goes to GitHub.

## Inputs

- The current branch. The base branch is `main`.
- Optional: the PR number, if the PR is open.

## Steps

1. Collect the facts. Use only these sources:
   - `git log --oneline main..HEAD`
   - `git diff --stat main...HEAD`
   - `git diff main...HEAD`
   - If the PR is open: `gh pr view <n> --json title,body,closingIssuesReferences` and `gh pr checks <n>`.
2. Copy `.github/pull_request_template.md` to `tmp-PR-content.md`. If `tmp-PR-content.md` exists, ask the developer before you replace it.
   - Add this heading as the first line: `# Draft PR Content`.
   - Add the line `Title: ` below the heading. Add an empty line after it.
   - If the PR is open, put its title from step 1 after `Title: `.
   - If the PR is open, `gh pr edit --body-file` replaces the existing body. Copy each existing section and link that the template does not have into `tmp-PR-content.md`. Get them from the `body` field of step 1. If you do not copy them, ask the developer to confirm before you discard them.
3. Write the `What` section. Obey `pr-what-prompt.md` in this folder. Use the diff from step 1 if the PR is not open. Add `Fixes #<n>` for each linked issue.
   - If the PR is not open, write the PR title after `Title: `. Use one short line that names the change.
   - If the PR is open, keep its title. Change the title only if the developer asks.
4. Write the release note. Obey `release-note-prompt.md` in this folder.
   - Put the note on the lines below the opening fence. Do not put text on the fence line.
   - Tell the developer which PR label the PR needs.
5. Write the `Verification` section. List only the commands that ran, with the result of each command. If no command ran, write `TODO: add the commands that you ran.` Do not write a result that you did not see.
6. Do not tick the boxes in the `Checklist` section. The developer ticks them.
7. Examine the text that you wrote. Do not examine the template text.
   - Apply the `asd-ste100` skill in STE-flavored mode.
   - Put the text that you wrote in a temporary file. Do not put the `Title:` line in it. Run `python3 .cursor/skills/asd-ste100/scripts/ste-lint.py --disable synonym-rotation <file>`.
   - Fix each hard violation.
8. Run the CI check on `tmp-PR-content.md`. Fix `tmp-PR-content.md` until the check passes:
   ```sh
   python3 - tmp-PR-content.md <<'PY'
   import re, sys
   body = open(sys.argv[1], newline="").read()
   m = re.search(r"^```release-note[ \t\r]*\n(.*?)^```[ \t\r]*$", body, re.DOTALL | re.IGNORECASE | re.MULTILINE)
   sys.exit(0 if m and m.group(1).strip() else "FAIL: no complete release-note block")
   PY
   ```
9. Ask the developer to approve the PR content:
   - Open `tmp-PR-content.md` in the editor. Use `code tmp-PR-content.md` or `cursor tmp-PR-content.md`. If neither command works, give the developer the file path.
   - Tell the developer to edit the file in the editor, or to ask for changes in the chat.
   - Tell the developer to write `approve` in the chat when the content is ready.
   - Do not continue until the developer writes `approve`.
10. Read `tmp-PR-content.md` again after the developer approves it. The developer can edit the file before approval. Do step 8 again.
    - Make sure that line 1 is `# Draft PR Content` and line 2 is the `Title:` line. Put them back if the developer moved them.
    - Make sure that the `Title:` line has a title. If it is empty, ask the developer for the title. Write the title after `Title: ` in `tmp-PR-content.md`.
11. Send the PR content to GitHub:
    - If the PR is not open, push the branch first. Ask for permission before you push. Obey the `mflux-pr` skill.
    - Run this block as one command. If the PR is open, replace the last line with `gh pr edit <n> --title "$title" --body-file tmp-PR-body.md`.
      ```sh
      title="$(awk 'NR <= 2 && sub(/^Title: */, "") { print; exit }' tmp-PR-content.md)"
      awk 'NR == 1 && /^# Draft PR Content$/ { next } NR == 2 && /^Title:/ { next } { print }' tmp-PR-content.md > tmp-PR-body.md
      gh pr create --title "$title" --body-file tmp-PR-body.md
      ```
    - The shell reads the title from the file. Do not type the title into the command. Backticks and `$` in a typed title run as shell code.
    - GitHub does not get the `# Draft PR Content` line or the `Title:` line. Other lines that start with `Title:` stay in the body.
12. Delete `tmp-PR-content.md` and `tmp-PR-body.md`.

## Rules

- Do not invent facts, numbers or test results.
- Do not change the code or the commits.
- Do not use `gh pr create --body` or `--fill`. GitHub must get the body that you checked, without the two draft lines.
- Keep the regex in step 8 the same as `.github/workflows/release-note.yml` and `_FENCE` in `src/mflux/release/release_notes.py`.
