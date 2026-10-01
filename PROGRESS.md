# Progress

## 2026-10-01: metrics and baseline methods M0-M2 (#3)

- **Done:** MSE/PSNR/SSIM metrics (proposal 3.6) and training-free baselines M0-M2 (proposal 3.4) behind a method registry, so experiments build methods by name from config.
- **Changed files:** `src/thermal_restore/metrics/{__init__,image}.py`, `src/thermal_restore/methods/{__init__,registry,m0_identity,m1_contrast,m2_spatial}.py`, `tests/test_metrics.py`, `tests/test_methods.py`.
- **Flow explained:** `get_method(name, **params)` validates the name and parameters, then returns a callable that checks `(T, H, W)` input. Registered names: `m0`, `hist_eq`, `clahe`, `gaussian`, `median`, `column`. MSE/PSNR use the whole array, SSIM is averaged per frame. `column` removes stripes by subtracting per-column offsets from a median-smoothed column profile.
- **Check:** `uv run pre-commit run --all-files`, `uv run ruff check .`, `uv run pytest` (29 passed).
