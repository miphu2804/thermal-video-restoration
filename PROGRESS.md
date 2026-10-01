# Progress

## 2026-10-01: CUDA-enabled detector environment and automatic device selection

- **Done:** Pinned the detector extra to `torch==2.13.0` and `torchvision==0.28.0`, configured the PyTorch CUDA 13.0 index for Windows/Linux, and added a runtime CUDA probe with automatic `cuda`/`cpu` selection.
- **Added:** `scripts/check_cuda.py` reports the selected device, CUDA runtime version, GPU name, capability, and probe errors; `--device cuda` now fails early with a useful diagnostic when CUDA is unavailable.
- **Environment check:** Installed `torch==2.13.0+cu130` and `torchvision==0.28.0+cu130`; CUDA probe selected `NVIDIA GeForce RTX 5060 Ti` successfully.
- **Check:** 52 tests passed, Ruff passed, and all pre-commit hooks passed. Detector training and inference were intentionally not run.

## 2026-10-01: Issue #4 detector foundation and optional torchvision backend

- **Done:** COCO-to-thermal TIFF adapter, backend-neutral detections, pure mAP@0.5 and mAP@[0.5:0.95] metrics, an optional torchvision MobileNetV3-Faster R-CNN backend, and user-run training/evaluation CLIs. Training and pretrained-weight downloads were intentionally not run.
- **Changed files:** `src/thermal_restore/{data/annotations,detection/types,detection/torchvision_backend,metrics/detection}.py`, package exports, `scripts/{train_detector,evaluate_detector}.py`, detection tests, `pyproject.toml`, `uv.lock`.
- **Flow explained:** COCO `.jpg` names are resolved to local thermal `.tiff` basenames; the train category map is preserved for validation and checkpoint loading; the detector repeats the normalized thermal channel to three channels for the pretrained torchvision backbone.
- **Check:** 49 tests passed, `uv run ruff check .`, pre-commit hooks passed, and all three real COCO splits loaded successfully.

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
