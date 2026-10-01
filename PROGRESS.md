# Progress

## 2026-10-01: FLIR ADAS loader, split by sequence, dataset download (#1)

- **Done:** FLIR ADAS v2 thermal loading (14-bit TIFF to float32 on the 0-255 scale), 70/15/15 split by video sequence, display helpers, a resumable download script, and a dataset check notebook. Dataset facts verified from the downloaded data are recorded in `docs/proposal.md` (3.2).
- **Changed files:** `src/thermal_restore/data/{__init__,loader,split,visualization}.py`, `scripts/download_dataset.sh`, `notebooks/01_dataset_check.ipynb`, `tests/test_{loader,split,visualization}.py`, `docs/proposal.md`, `README.md`, `AGENTS.md`, `.gitignore`, `pyproject.toml`, `uv.lock`.
- **Flow explained:** `list_sequences(root)` groups `video-<id>-frame-<n>-*.tiff` files by sequence in frame order; `split_by_sequence` shuffles sequence ids with a seed so no sequence lands in two sets. `.gitignore` now ignores only the top-level `/data/` so the `thermal_restore.data` package is tracked. `imagecodecs` is required because the TIFFs are LZW-compressed.
- **Check:** `uv run pre-commit run --all-files`, `uv run ruff check .`, `uv run pytest` (36 passed after merging `dev`); notebook executed top to bottom on the real data.

## 2026-10-01: metrics and baseline methods M0-M2 (#3)

- **Done:** MSE/PSNR/SSIM metrics (proposal 3.6) and training-free baselines M0-M2 (proposal 3.4) behind a method registry, so experiments build methods by name from config.
- **Changed files:** `src/thermal_restore/metrics/{__init__,image}.py`, `src/thermal_restore/methods/{__init__,registry,m0_identity,m1_contrast,m2_spatial}.py`, `tests/test_metrics.py`, `tests/test_methods.py`.
- **Flow explained:** `get_method(name, **params)` validates the name and parameters, then returns a callable that checks `(T, H, W)` input. Registered names: `m0`, `hist_eq`, `clahe`, `gaussian`, `median`, `column`. MSE/PSNR use the whole array, SSIM is averaged per frame. `column` removes stripes by subtracting per-column offsets from a median-smoothed column profile.
- **Check:** `uv run pre-commit run --all-files`, `uv run ruff check .`, `uv run pytest` (29 passed).

## 2026-10-01: degradation simulator on-the-fly (#2)

- **Done:** Pure function degradation operators (contrast compression, Gaussian blur, fixed pattern stripes, temporal stripes, Gaussian noise) in `src/thermal_restore/degradation/ops.py`, preset levels `light`/`medium`/`heavy` according to proposal 3.3 in `presets.py`, composable pipeline `degrade(frames, level, rng, steps=None)` in `pipeline.py`, unit tests in `tests/test_degradation.py`, and interactive demo notebook `notebooks/02_degradation_demo.ipynb`.
- **Changed files:** `src/thermal_restore/degradation/{__init__,ops,presets,pipeline}.py`, `tests/test_degradation.py`, `notebooks/02_degradation_demo.ipynb`, `scripts/download_dataset.py`, `PROGRESS.md`.
- **Flow explained:** All operators accept and return `(T, H, W)` `float32` arrays on the 0-255 scale. Fixed stripes use broadcasted column offsets constant across time frames; temporal stripes draw independent column offsets per frame; `steps` argument allows isolating specific degradation stages (e.g. for experiment E1).
- **Check:** `uv run pre-commit run --all-files`, `uv run ruff check .`, `uv run pytest` (47 passed).
