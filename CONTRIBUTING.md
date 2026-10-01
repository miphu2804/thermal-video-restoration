# Contributing

## Workflow

1. Pick an issue and assign yourself.
2. Branch off the latest `main`.
3. Commit, push, and open a Pull Request with `Closes #<issue-number>` in the description.
4. At least 1 approval is required; merge with **Squash and merge** (the branch is deleted automatically).

Never push directly to `main`.

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

Write the description in English, imperative mood, lowercase first word, no trailing period:

```
feat(degradation): add fixed and temporal stripe noise
fix(metrics): compute PSNR on 0-255 scale
```

Because PRs are squash-merged, the **PR title** becomes the commit on `main`, so it must follow the same format.

## Before opening a PR

```bash
uv sync
uv run ruff check .
uv run ruff format .
uv run pytest
```

## Data

Never commit datasets, checkpoints (`*.pt`), or outputs. The FLIR ADAS license forbids redistribution; keep it under `data/` (gitignored).
