# Contributing

## Workflow

1. Pick an issue and assign yourself.
2. Branch off the latest `dev`.
3. Commit, push, and open a Pull Request into `dev` with `Closes #<issue-number>` in the description.
4. At least 1 approval and passing checks are required; merge with **Squash and merge** (the branch is deleted automatically).

Never push directly, force-push, or manually merge into `dev` or `main`.

## Target branch

`dev` is the integration branch and `main` is the stable branch. Work happens on short-lived branches:

```text
feat/*, fix/*, exp/*, docs/*, chore/*  →  dev  →  main
```

- Open normal feature, fix, experiment, docs, and chore pull requests against `dev`.
- Once `dev` passes its checks, open a `dev` → `main` pull request for a release. Merge it with a **merge commit**, never squash: squashing disconnects the histories of `dev` and `main` and forces a manual resync.
- Start hotfixes from `main`, open the pull request against `main`, then bring the same fix back into `dev`.
- Do not introduce other long-lived branches without an explicit workflow change.

Create a branch from `dev`:

```bash
git fetch origin
git switch dev
git pull --ff-only
git switch -c feat/<issue-number>-<short-description>
```

## Language

- **English:** code, comments, docstrings, commit messages, branch names, PR titles, and every Markdown file.
- **Vietnamese:** only `docs/proposal.md` (the reading copy of the proposal) and issue/PR descriptions.

## Branch names

`<type>/<issue-number>-<short-description>`, lowercase, words joined by `-`.

```
feat/002-degradation-simulator
fix/003-psnr-range
docs/001-flir-stats
```

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>
```

| Type | Use for |
|---|---|
| `feat` | New functionality (method, metric, loader, ...) |
| `fix` | Bug fix |
| `exp` | Experiments, notebooks, run configs |
| `docs` | Documentation |
| `test` | Adding or changing tests |
| `refactor` | Restructuring without behavior change |
| `chore` | Dependencies, tooling config |

Suggested scopes: `data`, `degradation`, `methods`, `metrics`, `detection`.

Keep the whole message at most 72 characters. Describe the action taken, not the resulting state, in English, imperative mood, lowercase first word, no trailing period. Keep each commit to one logical change and do not mention AI generation or co-authorship unless explicitly requested:

```
feat(degradation): add fixed and temporal stripe noise
fix(metrics): compute PSNR on 0-255 scale
```

Because PRs into `dev` are squash-merged, the **PR title** becomes the commit on `dev`, so it must follow the same format.

## Pull request description

Use this structure (the description may be in Vietnamese):

```markdown
## Summary
What changed and why.

## Changes
Important implementation changes.

## Validation
Tests, checks, or manual verification performed.

## Risks
Known risks, compatibility concerns, or None.
```

Identify related docs (e.g. the proposal section), assumptions, and anything left unverified.

## Review, rebase, and merge

- Fetch the latest remote state before opening or finalizing a pull request.
- Rebase a personal branch onto the latest `dev` when appropriate.
- If a pushed branch is rebased, use `--force-with-lease`, never `--force`.
- Do not rebase or force-push shared branches.
- Do not bypass review, CI, or branch protection.

## Before opening a PR

```bash
uv sync
uv run pre-commit install   # once: isort + black on every commit (.py and .ipynb)
uv run pre-commit run --all-files
uv run ruff check .
uv run pytest
```

## Data

Never commit datasets, checkpoints (`*.pt`), or outputs. The FLIR ADAS license forbids redistribution; keep it under `data/` (gitignored).

## Progress log

After each completed substantive change, add an entry at the top of `PROGRESS.md` (newest first; it is a log, not the source of truth for scope). Keep it short:

- **Done:** work completed;
- **Changed files:** files created, modified, or deleted;
- **Flow explained:** behavior or flow that changed;
- **Check:** checks that were run, if any.

## Blockers

Create or update `BLOCKERS.md` only while an unresolved blocker exists; do not keep an empty file. Each blocker includes:

- **Status:** `Open` or `Resolved`;
- **Blocked by:** the cause or dependency;
- **Impact:** the affected work;
- **Next action:** the smallest next action.

In the pull request or issue, also record the symptom and the last verified boundary. Do not mark the work complete while a blocker remains. If a command cannot run, record the last verified boundary instead of substituting an unverified command.
