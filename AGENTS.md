# AGENTS.md

## Project

Restoring degraded thermal infrared video before object detection. The full specification is in `docs/proposal.md` (Vietnamese) — read the relevant section before changing code (3.3 degradation simulator, 3.4 methods M0–M7, 3.6 experiments and metrics).

## Environment

- Python 3.11, managed with `uv`. Add dependencies with `uv add`; never edit `uv.lock` by hand.
- Formatting: isort + black (also on notebooks) via pre-commit. Checks: `uv run pre-commit run --all-files && uv run ruff check . && uv run pytest`.

## Layout

```
src/thermal_restore/
  data/         # FLIR ADAS 14-bit loading (loader), split by sequence, display helpers
  degradation/  # degradation simulator (proposal 3.3)
  methods/      # restoration methods M0–M7, registry + factory
  metrics/      # MSE, PSNR, SSIM, mAP
  detection/    # frozen object detector wrapper
tests/  notebooks/  data/ (gitignored)
configs/        # experiment YAML (method names, params, seeds); no secrets
```

## Design

- Each package exposes small, reusable modules; import from the package, not from notebooks.
- Prefer pure functions (numpy array in, array out); file I/O stays in `data/` or scripts.
- Frames are `float32` arrays on the 8-bit-equivalent scale (0–255); sequences have shape `(T, H, W)`.
- `degradation/`: one pure function per step, composed by a single `degrade(frames, level, rng)`.
- `methods/`: every method has the signature `(frames: np.ndarray, **params) -> np.ndarray` and is registered by name; experiments build methods through `get_method(name, **params)` so E1–E4 can loop over names from config.
- Use a class only when it owns state (e.g. the detector model, the M7 network).

## Conventions

- Degradation parameters use the 8-bit-equivalent scale (0–255), as in the proposal.
- Anything random takes a `seed` or `np.random.Generator` for reproducibility.
- Split data by video sequence, never by frame (avoids leakage).
- Code, comments, docstrings (Google style), commit messages, branch names, PR titles, and Markdown files are in **English**. Only `docs/proposal.md` and issue/PR descriptions are in Vietnamese.
- Never commit data, checkpoints, or outputs.
- Secrets (if ever needed) come from environment variables or a gitignored `.env`, never from `configs/`.

## Git

See `CONTRIBUTING.md`: branch `<type>/<issue-number>-<short-description>`, Conventional Commits, PRs into `main` need 1 approval, squash merge.
