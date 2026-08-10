# Migrating MFlux to it's own Organization

Here’s a practical migration checklist for moving `filipstrand/mflux` from a personal account to an organization with minimal disruption.

## 1) Pre-migration decisions (do this first)
- Pick org name and create org.
- Decide who will be **Org Owners** (at least 2).
- Decide repo visibility (public/private) after transfer.
- Decide default branch name and branch protection policy.
- Decide who manages releases/packages/actions billing.

## 2) Access + security setup in the new org
- Enable 2FA requirement for org members (recommended).
- Create teams (`core`, `maintainers`, `triage`) and map permissions.
- Configure org-level security settings:
  - Branch protection defaults / rulesets
  - Dependabot + secret scanning (if available)
  - Code owners expectations

## 3) Repo hygiene before transfer
- Clean up admin access on the personal repo.
- Merge/close stale PRs if possible.
- Confirm `README`, `CONTRIBUTING`, `CODE_OF_CONDUCT`, `SECURITY.md`, `LICENSE` are present/up to date.
- Add/verify `CODEOWNERS`.
- Snapshot critical settings (screenshots/export):
  - Branch protections/rulesets
  - Webhooks
  - Actions settings
  - Environments + required reviewers
  - Secrets/variables list (names only; values can’t be exported)

## 4) CI/CD and integrations audit (important)
- List all GitHub Actions secrets and variables (repo + environment + org-level dependencies).
- Check external integrations:
  - PyPI publish tokens
  - Cloud creds
  - Discord/Slack webhooks
  - Any app installations
- Check deploy keys, GitHub Apps, PAT-based automations.
- Verify who owns package namespaces (if using GHCR/PyPI).
- Plan secret re-entry in org/repo after transfer (many tokens should be rotated anyway).

## 5) Communicate freeze window
- Announce a short maintenance window to contributors.
- Ask contributors to pause merges during transfer.
- Post a pinned issue/discussion with timeline and expectations.

## 6) Perform the transfer
- From personal repo settings, transfer `filipstrand/mflux` to new org.
- Keep same repo name (`mflux`) unless there is a naming conflict.
- Verify transfer completion and permissions immediately.

## 7) Immediate post-transfer validation
- Confirm old URLs redirect to new repo.
- Validate:
  - Issues, PRs, labels, milestones
  - Releases, tags, branches
  - Wiki, Discussions, Projects linkage
- Re-check branch protections/rulesets (sometimes need re-application/tuning).
- Re-enable or fix any broken webhooks/integrations.
- Reconfigure Actions secrets/variables/environments as needed.
- Run CI on:
  - default branch
  - a sample PR
  - release workflow (dry run if possible)

## 8) Update ecosystem references
- Update badges, docs, and clone URLs.
- Update package metadata/homepage links (`pyproject.toml`, setup metadata, docs site).
- Update links in README, contribution docs, and pinned discussions/issues.
- If website/docs mention old owner path, replace it.

## 9) Protect governance continuity
- Add at least 2–3 maintainers as org owners/admins.
- Use teams instead of direct user permissions for maintainability.
- Document decision-making and release ownership in `GOVERNANCE.md` (optional but useful).

## 10) Announce completion
- Publish a short “migration complete” announcement:
  - New canonical repo URL
  - Any contributor actions needed (`git remote set-url`)
  - Confirmation that old links redirect

---

## Quick command contributors may need
```bash
git remote set-url origin git@github.com:<new-org>/mflux.git
# or
git remote set-url origin https://github.com/<new-org>/mflux.git
```

## Common gotchas
- Lost/broken Actions secrets after transfer.
- Branch protection drift.
- External services still pointing at old webhook endpoints.
- Package publish permissions tied to old owner namespace.
- Only one org owner (single point of failure).

If you want, I can turn this into a **step-by-step runbook with owner assignments and a 60-minute migration timeline** you can paste into an issue.
