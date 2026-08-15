---
name: mflux-release
description: Prepare a release in mflux (version bump, changelog, contributors, uv lock) without tagging/publishing. Use when preparing a release branch or release PR.
---
# mflux release prep

Releases are prepared in-repo; tagging/publishing is handled by GitHub Actions.

## When to Use

- You’re preparing a release PR.
- You need to bump version and update the changelog/lockfile correctly.

## Instructions

- Checklist:
  - Bump version in `pyproject.toml`
  - Review commits since last release tag:
    - `git log --oneline v.<last-version>..HEAD`
  - Browse relevant file contents if needed to clarify changes
  - Weigh commit impact: some are minor; skim earlier changelog entries to gauge what's worth reporting vs tiny fixes
  - Add a descriptive entry to `CHANGELOG.md` based on those commits
  - Always add a `### 👩‍💻 Contributors` section to the new changelog entry
  - Source contributor names from merged GitHub PR authors for the changes included in the release, and format them as `@handle`
  - Prefer one release commit for the release prep work on the branch
  - Name that commit `Release <version>`
  - Prefer GitHub data over local git author names:
    - Use `gh pr list --state merged` / related `gh` queries when available
    - Do not assume `gh` is installed; if it is unavailable, inspect GitHub on the web instead
    - General pages to check on the web:
      - Closed PRs: `https://github.com/<owner>/<repo>/pulls?q=is%3Apr+is%3Aclosed`
      - Compare view for release range: `https://github.com/<owner>/<repo>/compare/v.<last-version>...HEAD`
      - Specific PR page when you know the PR number: `https://github.com/<owner>/<repo>/pull/<number>`
    - Map included changes to merged PR authors from those pages
  - Do not invent handles, and do not treat local-only commits as contributors unless the user explicitly wants that
  - Update lockfile: `uv lock`
  - Sanity checks (optional unless requested): `just test-fast`, `just build`
  - Manual checks (optional): if the release includes CLI/callback/image-path changes, consider running the `mflux-manual-testing` skill to exercise the touched commands and visually review outputs.
- Do not tag releases locally unless explicitly requested (normally handled by CI).

