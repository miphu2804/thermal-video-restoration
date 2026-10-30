# Progress

## 2026-10-17: input preprocessing candidates for inference

- **Done:** Added opt-in point, spatial, Fourier and wavelet preprocessing methods in `src/thermal_restore/methods/m3_preprocessing.py`: linear point operation, full-scale/percentile stretch, logarithmic transform, arithmetic/geometric/harmonic/contraharmonic means, Fourier low-pass/notch filters and soft-threshold wavelet denoising. All preserve finite `float32` frames on the 0-255 scale and are built through the existing registry.
- **Evaluation:** Added `scripts/evaluate_preprocessing.py` and `configs/inference_preprocessing.json`. It applies candidates directly to the same clean labelled frames with one frozen checkpoint, preserving source normalization, image IDs and AP/quality/timing reports. Parameters must be selected on validation before test.
- **Inference:** Added `--preprocess-pipeline` to `scripts/infer_realtime.py` for direct video/camera inference. It is mutually exclusive with the synthetic A/B/C demo and records the selected stages in optional JSON reports. Temporal integration, frame differencing and motion compensation remain out of this single-frame option because they require sequence state and alignment.
- **Visual comparison:** Added `--compare-preprocessing` to show original+detector beside preprocessed+detector for the same source frame. Both branches use the same checkpoint, threshold and normalization; each frame requires two inference passes. The shared source display range keeps contrast changes visible.
- **Bounded checks:** Four-image CUDA preprocessing smoke: M0 reproduced clean A; FSHS/log caused a large distribution shift; mean/Fourier/wavelet candidates preserved high image similarity but did not establish a winner. Two-frame downloaded-MP4 wavelet inference completed on CUDA. These are wiring checks, not full validation conclusions.
- **Validation:** Added 18 preprocessing tests and the two-panel rendering test; the full suite passes 122 tests. Ruff and the formatted-file pre-commit checks pass. Full candidate evaluation remains user-run on validation.

## 2026-10-04: frozen detector A/B/C demo and benchmark

- **Done:** Added `--compare-abc` to interactive inference and a separate `scripts/evaluate_abc.py` validation/test entry point. Both reuse one checkpoint and registry-configured M0-M2 restoration chains from `configs/abc_baselines.json`. A is the source reference, B is simulated degradation, and every C pipeline receives the exact same B. No training or detector-weight edits occur. M3-M7 and motion-compensated temporal restoration are outside this initial baseline implementation.
- **Reproducibility:** Added a pure frame-keyed streaming degradation helper and optional fixed-offset injection into the existing simulator, preserving its default API/behavior. Fixed stripes are keyed by sequence, other noise by frame and seed, independently of sampling/batch/method order. Rendering uses one A-derived display range across panels and never changes inference tensors. The shared Tk window now lives in the package rather than a script.
- **Reporting:** JSON/CSV/Markdown record mAP@0.5, mAP@[0.5:0.95], per-category AP, pixel-weighted MSE/global PSNR, frame-averaged SSIM, C-minus-B AP deltas, separate stage timings and noise-seed aggregates. Original image/annotation IDs, source-policy exclusions, checkpoint/configuration metadata and optional COCO prediction JSONL preserve auditability. A is evaluated once; existing output directories/results are protected from overwrite.
- **Source scope:** The quantitative default uses raw14 references only. On the local test mirror this excludes 1,598 uint8 TIFFs, leaving 2,151 frames; A must be recomputed on the same subset, not compared with old mixed-test mAP. Explicit mixed-source experiments are labelled separately. Downloaded MP4 A/B/C demos are unlabelled compressed-source comparisons, not radiometric benchmarks.
- **Bounded validation:** CUDA smoke runs completed on four validation frames with cached inputs and all configured pipelines, plus three-frame raw-TIFF and downloaded-MP4 demos with MP4 export/readback. M0 reproduced B exactly. An isolated contrast-only smoke retained detections while CLAHE reduced AP; full medium degradation reached a detection floor on the tiny sample. These are plumbing/diagnostic checks, not full-dataset results or evidence of a winning restoration method. Full runs and parameter selection remain user-run on validation, followed by a fixed test configuration.
- **Checks:** Added seven pure-helper tests and mocked end-to-end benchmark cases for full-policy and seeded-sample scopes; the full suite passes 103 tests. Pre-commit and Ruff passed, the exported comparison frames were inspected, and the refactored A/B/C Tk window passed an image/pause test in hidden mode. Generated artifacts remain under ignored `outputs/`.

## 2026-10-04: interactive detector-only inference

- **Done:** Added `scripts/infer_realtime.py` with a Tk window, bounding boxes, category names, confidence, single-frame inference latency, model/loop FPS, pause/resume and quit controls. Sources include one ordered FLIR TIFF sequence, a TIFF file, or explicitly allowed uint8 AGC thermal video/camera. Lazy sources release capture resources on close/error; multiple FLIR sequences require explicit selection. No changes to training, checkpoint weights or mAP evaluation.
- **Input contract:** Raw TIFFs pass through the canonical 14-bit loader and checkpointed fixed normalization. Allowed uint8 inputs use exactly the evaluator's source-aware statistics. Contrast stretching is display-only. RGB/pseudocolor capture is not equivalent to thermal data; camera uint16 output is refused because its sensor calibration is unknown.
- **Timing:** Best-effort sequential playback, not a guaranteed 30 FPS or hard real-time system. Reports distinguish inference/transfer time, total processing and paced loop throughput. Model loading is excluded; first-frame warm-up is included. Camera-driver buffering may add latency. Optional MP4 exports are fixed-FPS visualizations rather than timestamp-accurate recordings; existing exports are not overwritten.
- **Validation:** Five-frame CUDA smoke runs completed for both uint16 and uint8 using the existing ResNet50 checkpoint. Annotated MP4 writing/readback succeeded and a hidden Tk PPM image check succeeded. Pre-commit, Ruff and all 94 tests passed. Full interactive playback and physical thermal camera acquisition remain user-run; the main training run was not started. Added nine synthetic tests for source selection/order, raw scale, uint8 policy, capture cleanup and display isolation.

## 2026-10-04: source annotation and focus-class audit

- **Done:** Added a read-only annotation audit and user-reusable `scripts/audit_detector_data.py`: exhaustive TIFF/COCO geometry checks, exact duplicate boxes, invalid/zero-area and out-of-bounds boxes, sequence overlap, class support/size statistics, and seeded paired bike/motor crop sheets. Preview sequence order is shuffled reproducibly, avoiding alphabetical sampling bias. Candidate coverage distinguishes low confidence, class confusion, poor localization and missing candidates; it is not one-to-one recall or AP.
- **Real-data result:** All 15,635 image dimensions match COCO; no sequence IDs overlap between splits. The only structural issues are 2 exact duplicate training boxes, 8 validation boxes and 1 test box. No labels or pixels were changed. Training bike/motor median box sizes are 21x30 / 34x40 pixels, versus validation 28x38.5 / 39x43 and test 11x14 / 21x25. Motor has only 1,116 training frame instances, and 55 validation instances across 6 sequences.
- **Existing test dump:** Reused the fixed ResNet50 checkpoint's predictions, without rerunning full test inference. Of 113 bike GT instances, 106 have no matching candidate under the audit rules and 7 have same-class IoU >= 0.5 below confidence 0.3. Of 3,314 motor instances: 142 have same-class IoU >= 0.5 at confidence >= 0.3, 383 only below 0.3, 356 overlap another class, 204 have poor same-class localization and 2,229 have no matching candidate. Counts describe this checkpoint and repeated frames of one sequence, not independent objects or benchmark recall.
- **Interpretation:** Small objects, occlusion and source/domain differences are visible in the crop samples. Exhaustive geometry validation plus 48 visual samples cannot certify semantic correctness or establish a single cause of the validation/test gap. Keep original labels and the existing checkpoint; select any new training settings only on train/validation, not on this test audit. Main training remains user-run.
- **Validation:** Full local audit completed on CUDA with train/validation inference restricted to seeded preview frames. Generated JSON, Markdown and six crop sheets stay under ignored `outputs/detector/data_audit/`. Added three synthetic unit tests; the full suite passes 85 tests.

## 2026-10-04: dtype, sequence and visual detector diagnostics

- **Finding:** ResNet50-FPN-v2 reached validation mAP@0.5 0.4963 and test mAP@0.5 0.2201. The mirror test split mixes 2,151 uint16 frames with 1,598 uint8 frames; source-aware uint8 normalization differs from the checkpoint's raw-thermal statistics.
- **Done:** Added optional `--diagnostics-dir` evaluation exports: recomputed AP by dtype and sequence, class support and small-box fractions, actual source normalization, original-ID predictions, and seeded ground-truth/prediction comparison PNGs with bike/motor focus. Preview confidence thresholds never change the predictions used for mAP.
- **Validation scope:** `--max-images` selects a seeded smoke sample and labels outputs explicitly as incomplete. Full diagnostic evaluation remains a user-run command. Group differences are descriptive, not proof of a dtype effect or grounds for selecting hyperparameters on test.
- **Checked:** A 16-image CUDA smoke run covered both dtypes, exported nine group reports and eight comparison PNGs; full Ruff and 82 tests passed.

## 2026-10-04: detector progress and ETA reporting

- **Done:** Added real in-place `tqdm` progress bars to detector training, validation, and standalone evaluation. The bars show completed/total batches, percentage, elapsed time, ETA, throughput, and periodically refreshed loss, learning rates, or prediction count. Epoch summaries retain epoch duration, total run duration, and estimated remaining run time.
- **Usage:** The bar advances every batch. Use `--progress-interval N` to control how often its metric postfix is refreshed (100 by default), or `--progress-interval 0` to disable it. Standalone evaluation announces the separate mAP calculation stage after inference.
- **Check:** A short detector smoke run verifies the progress bar without validation or checkpoint writes; full training remains user-run.

## 2026-10-04: optional stronger detector backbone

- **Finding:** Thermal standardization raised the MobileNet validation mAP@0.5 from 0.1488 to 0.1810, with the best checkpoint at epoch 18. The five principal road-user categories average about 0.372 AP@0.5, while rare and very small categories continue to dominate the lower all-class macro score.
- **Done:** Added an optional Faster R-CNN ResNet50-FPN-v2 backend with the same category mapping, fixed thermal normalization, differential learning rates, warm-up, finite guards, and checkpoint contract. Its five FPN levels use one FLIR-oriented anchor size each (8/16/32/64/128), preserving the pretrained RPN head shape.
- **Scope:** MobileNet remains the default small detector required by the proposal. ResNet50-FPN-v2 is a materially larger comparison candidate; if selected on validation, freeze that one checkpoint for every A/B/C experiment and report the added runtime/cost.

## 2026-10-03: stable full-backbone detector fine-tuning

- **Finding:** The first thermal-standardized run used learning rate 0.005 for the newly initialized detector head and every unfrozen pretrained backbone parameter. It diverged to non-finite losses during epoch 1, produced no predictions, and wrote an unusable checkpoint.
- **Done:** Added a separate 0.0005 backbone learning rate, 1,000-iteration linear warm-up, gradient clipping, named loss-component logging, and finite loss/gradient guards. Non-finite training now exits immediately before the optimizer step, validation, or checkpoint write. Cosine decay preserves the detector/backbone learning-rate ratio.
- **Validation:** A 1,200-batch CUDA smoke test crossed the entire warm-up and remained finite: total loss 0.8052, classifier 0.2915, box regression 0.1782, objectness 0.1843, and RPN box regression 0.1513. Diagnostic batch-limited runs intentionally skip validation and checkpointing.

## 2026-10-03: fixed thermal input standardization

- **Finding:** The high-resolution MobileNet run improved validation mAP@0.5 from the corrected legacy baseline 0.0823 to 0.1488, but deterministic train-cache sampling found a mean of about 113.25 and standard deviation of only 5.66 on the 0-255 scale. The legacy ImageNet transform therefore presented the pretrained backbone with nearly flat inputs.
- **Done:** Added fixed mean/std standardization estimated only from the clean training split, full MobileNet backbone fine-tuning, checkpointed normalization statistics, and source-aware display statistics for explicitly allowed uint8 mirror frames. This fixed transform preserves contrast degradation instead of cancelling it with per-frame stretching.
- **Reporting:** The primary all-class validation mAP remains 0.1488 for the completed high-resolution checkpoint. Its AP@0.5 is 0.4730 for car, 0.4135 for motor, 0.2774 for bus, and 0.2478 for person, while the macro average is strongly reduced by classes with as few as 3-6 validation instances.
- **Pending:** Full retraining with thermal standardization remains a user-run operation. Keep reporting the all-class metric and per-category AP rather than removing rare classes from the primary score.

## 2026-10-03: FLIR small-object detector configuration

- **Finding:** The first cached SGD run plateaued at validation mAP@0.5 0.0711. Half of the 175,040 training boxes are at most about 15x21 pixels at the original 512x640 resolution, while the previous 320-pixel transform reduced them further and the smallest RPN anchor was 32 pixels.
- **Done:** Added the higher-resolution pretrained MobileNetV3-FPN variant, 512x640 detector input, FLIR-oriented 8/16/32/64/128 anchors, a gradual cosine learning-rate schedule, and checkpointed model configuration. Legacy checkpoints still load with their original 320-pixel/default-anchor configuration.
- **Metric correction:** Validation and evaluation now retain low-confidence detections down to 0.001 before constructing the precision-recall curve. The previous 0.25 cutoff understated AP and could select the wrong checkpoint.
- **Pending:** Full high-resolution retraining remains a user-run operation. The previous checkpoint measured mAP@0.5 0.0801 on validation when rescored down to the backend's former 0.05 internal floor, confirming that the 0.25 cutoff discarded useful recall but was not the only quality limitation.

## 2026-10-03: memory-mapped detector frame cache

- **Done:** Added a reusable per-split NumPy memory-map cache for detector frames, a cache-building CLI, manifest validation against COCO image order and dimensions, constant-time annotation lookup by image ID, Windows DataLoader worker safety, persistent train/validation workers, and non-blocking CUDA transfers.
- **Data contract:** Cached frames remain single-channel `float32` on the 0-255 scale. Random horizontal flips and three-channel conversion stay on-the-fly, so caching does not freeze augmentation or triple storage. Test `uint8` mirror sequences require the same explicit policy when building and reading a cache.
- **Storage estimate from real data:** train 13.11 GiB, validation 1.40 GiB, test 4.58 GiB. All cache files remain below the ignored top-level `data/` directory.
- **Check:** 71 tests passed, including synthetic cache build/load/policy tests. Full cache generation and detector training were intentionally left as user-run operations.

## 2026-10-03: detector training and mixed-bit-depth evaluation improvements

- **Done:** Added the standard Faster R-CNN SGD optimizer with a step scheduler, training-only horizontal flip augmentation with box updates, explicit `uint8` handling for mirror sequences, per-category AP reporting, and checkpoint loading without a redundant backbone download.
- **Why:** The initial detector used AdamW with a very small learning rate, while 1,598 test frames across two sequences are contrast-stretched `uint8` TIFFs. Treating those as 14-bit produces values around 0–4 instead of 0–255 and contaminates the test result.
- **Usage:** The evaluator keeps strict 14-bit behavior by default; pass `--allow-uint8` only when intentionally evaluating the mixed-bit-depth test split. Training uses SGD, scheduler, and horizontal flips by default; use `--optimizer adamw` or `--no-horizontal-flip` for controlled comparisons.
- **Check:** 67 tests passed, Ruff passed, and all pre-commit hooks passed. Full retraining and test inference were intentionally left for the user.

## 2026-10-03: reject non-uint16 frames in load_frame (#14)

- **Done:** `load_frame` raises a `ValueError` naming the file and dtype for any TIFF that is not `uint16`, instead of silently rescaling it. Choosing how to handle the two 8-bit sequences (`video-ZAtDSNuZZjkZFvMAo`, `video-t3f7QC8hZr6zYXpEZ`) is still open.
- **Changed files:** `src/thermal_restore/data/loader.py`, `tests/test_loader.py`, `PROGRESS.md`.
- **Flow explained:** `load_frame` checks `raw.dtype` before scaling by 255/16383; contrast-stretched `uint8` frames cannot be converted back to 14-bit counts, so they are refused. Loading either 8-bit sequence now fails loudly.
- **Check:** `uv run pre-commit run --all-files`, `uv run ruff check .`, `uv run pytest` (50 passed).

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

## 2026-10-01: degradation simulator on-the-fly (#2)

- **Done:** Pure function degradation operators (contrast compression, Gaussian blur, fixed pattern stripes, temporal stripes, Gaussian noise) in `src/thermal_restore/degradation/ops.py`, preset levels `light`/`medium`/`heavy` according to proposal 3.3 in `presets.py`, composable pipeline `degrade(frames, level, rng, steps=None)` in `pipeline.py`, unit tests in `tests/test_degradation.py`, and interactive demo notebook `notebooks/02_degradation_demo.ipynb`.
- **Changed files:** `src/thermal_restore/degradation/{__init__,ops,presets,pipeline}.py`, `tests/test_degradation.py`, `notebooks/02_degradation_demo.ipynb`, `PROGRESS.md`.
- **Flow explained:** All operators accept and return `(T, H, W)` `float32` arrays on the 0-255 scale. Fixed stripes use broadcasted column offsets constant across time frames; temporal stripes draw independent column offsets per frame; `steps` argument allows isolating specific degradation stages (e.g. for experiment E1).
- **Check:** `uv run pre-commit run --all-files`, `uv run ruff check .`, `uv run pytest` (47 passed).
