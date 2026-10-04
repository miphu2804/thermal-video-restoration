# thermal-video-restoration

Restoring degraded thermal infrared video before object detection (IMP302m).
The project compares clean, synthetically degraded and restored thermal frames
using one frozen detector. Proposal (Vietnamese): [docs/proposal.md](docs/proposal.md).

For team members who only need to run the existing model, start with
[Team quick start: model placement and running](#team-quick-start-model-placement-and-running)
at the end of this README. No retraining is required.

## Setup

Use Python 3.11 and `uv`. For data, restoration and development only:

```powershell
uv sync
uv run pre-commit install
```

For detector training, inference or evaluation, install the optional detector
dependencies instead:

```powershell
uv sync --extra detector
```

The project pins `torch==2.13.0` and `torchvision==0.28.0`; the configured
Windows/Linux package source is the CUDA 13.0 wheel index. Do not manually edit
`uv.lock` or replace these packages with a different build for the team run.
The commands below use `uv run --extra detector` to request this extra explicitly.

Dataset download requires your own Kaggle credentials at `~/.kaggle/kaggle.json`
and downloads approximately 4.7 GB into `data/`:

```bash
bash scripts/download_dataset.sh 3
```

On Windows, run this download command in Git Bash or a configured WSL
environment with Bash, Python/uv and access to the credentials. The other
commands in this README run directly in PowerShell. If PowerShell's `bash`
reports that `/bin/bash` is missing, use Git Bash or configure WSL first.
Never share or commit Kaggle credentials. Keep the FLIR dataset out of Git and
do not redistribute it without confirmed permission.

Checks before a PR:

```bash
uv run pre-commit run --all-files && uv run ruff check . && uv run pytest
```

## Layout

```
src/thermal_restore/
  data/         # FLIR ADAS 14-bit loading (loader), split by sequence, display helpers
  degradation/  # degradation simulator (proposal 3.3)
  methods/      # restoration methods M0–M7
  metrics/      # MSE, PSNR, SSIM, mAP
  detection/    # frozen object detector wrapper
  experiments/  # shared A/B/C degradation and restoration helpers
configs/        # experiment configs, including JSON A/B/C baselines
notebooks/      # exploration and dataset checks
scripts/        # download, training, inference and evaluation entry points
tests/
data/           # local dataset, optional caches and demo videos; not committed
outputs/        # local checkpoints, reports and recordings; not committed
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the git workflow.

## Interactive thermal detector inference

This project studies restoration of degraded thermal video, then measures
whether a fixed detector recovers detection quality. The live demo below is
**detector-only** unless `--compare-abc` is supplied. It is not tracking or a
replacement for labelled mAP evaluation.
It loads your existing checkpoint without retraining or downloading weights.

Run from the repository root (one line):

```powershell
uv run --extra detector python scripts/infer_realtime.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --source data/FLIR_ADAS_v2/video_thermal_test --sequence video-4FRnNpmSmwktFJKjg --device auto --score-threshold 0.3 --target-fps 30
```

Press **Space** to pause/resume, **Q** or **Esc** to quit, or close the window.
The Tk window works with the existing headless OpenCV dependency; no second
OpenCV package is needed. Python must include Tcl/Tk for graphical mode.

- TIFFs are read lazily in source frame order through the canonical 14-bit
  loader. Directories containing multiple FLIR sequences require `--sequence`.
- Checkpoint mean/std and dtype-specific normalization match evaluation.
  Display contrast stretching never feeds back into the model.
- For mirror uint8 TIFF sequences, explicitly add `--allow-uint8`.
- A local grayscale/AGC thermal video uses `--source path/to/thermal.mp4
  --allow-uint8` without `--sequence`. A thermal camera exposed through
  OpenCV uses `--camera 0 --allow-uint8` instead of `--source`. RGB/pseudocolor
  webcam feeds are out of the training domain; grayscale conversion does not
  turn them into radiometric thermal data. Camera uint16 output is rejected
  because sensor calibration/depth is unknown; use supported TIFFs instead.
- FPS is measured, not guaranteed. `model FPS` includes single-frame inference
  and device transfers; `loop FPS` also includes input decoding, preprocessing,
  rendering and playback pacing. The 30 FPS setting is only a cap. Slow
  inference slows sequential playback; camera-driver buffering can add latency.
  First-frame warm-up is included and model loading is excluded.
- `--target-fps 0 --no-display --max-frames 20` performs a bounded throughput
  smoke check. `--report outputs/detector/live_run.json` saves configuration
  and timings; `--output-video outputs/detector/live_run.mp4` optionally records
  the annotated display at the target FPS (30 when the cap is disabled).
  Saved video is not a real-time timestamp recording; no frames are skipped to
  catch up and filename gaps are not replayed as missing time. Existing export
  files are never overwritten. Data, reports and recordings belong under the
  ignored `data/` or `outputs/` directories.

## Frozen detector A/B/C comparison

The demo and evaluator share the same degradation/restoration helpers:

- **A:** unchanged source reference.
- **B:** simulated contrast compression, blur, fixed/temporal stripes and noise.
- **C:** restoration applied to that exact B frame, not to a new noise draw.

One checkpoint and the same source-specific normalization are used throughout.
No detector training takes place. Initial M0-M2 chains and parameters are in
`configs/abc_baselines.json`, built through the method registry. They are
starting baselines, not validated optimal restoration settings or implementations
of M3-M7. `m0` is an identity control and must reproduce B. Other configured
chains include CLAHE after filtering, following proposal 3.4. OpenCV CLAHE
clips/rounds to uint8 as documented by the existing method implementation;
its output is not automatically re-normalized to the clean thermal histogram.
Better-looking enhanced images can have worse PSNR or detection AP.

### Three-panel interactive demo

```powershell
uv run --extra detector python scripts/infer_realtime.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --source data/FLIR_ADAS_v2/video_thermal_test --sequence video-4FRnNpmSmwktFJKjg --device auto --compare-abc --pipeline column_median_clahe --level medium --seed 0 --score-threshold 0.3 --target-fps 30
```

This chain is an experimental baseline, not a recommended enhancement; see the
validation findings below before interpreting its output as an improvement.
Use Space to pause and Q/Esc to quit. Each source frame requires three inference
passes, so playback can be slower than detector-only mode. The panels share A's
percentile display range for raw TIFF, or a common 0-255 range for AGC video.
Independent display stretching would conceal contrast degradation, so it is
not used. Panel resizing is display-only; model input sizes remain checkpointed.

For a downloaded 8-bit thermal MP4, replace the TIFF source/sequence arguments
with `--source "path/to/video.mp4" --allow-uint8`. A is then the compressed
source-video reference, not proven clean radiometric data. The demo has no
ground-truth mAP and does not replace quantitative FLIR evaluation.

### Labelled validation experiment

```powershell
uv run --extra detector python scripts/evaluate_abc.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --data-root data/FLIR_ADAS_v2 --split val --device auto --batch-size 2 --levels medium --seeds 0 --progress-interval 10 --output-dir outputs/detector/abc_val
```

The example reads TIFFs directly: no shared preprocessing cache is required.
If you have already built a compatible cache, optionally add
`--cache-root data/cache/detector_float32_v1`.
The output directory must be new or empty; existing results are never
overwritten. `metrics.json`, `metrics.csv` and `report.md` record mAP@0.5,
mAP@[0.5:0.95], per-class AP, MSE, PSNR, SSIM, C-minus-B AP changes and stage
timings. JSON also stores checkpoint SHA256, original image/annotation IDs,
source exclusions, normalization, stage parameters and seeds. Optional
`--save-predictions` exports original-ID COCO prediction JSONL for every condition.

- Use `--max-images 4` for a seeded **smoke sample**, not a reportable full result.
  `--sample-seed` chooses that sample independently of noise seeds.
- Use `--levels light medium heavy --seeds 0 1 2` for repeated noise experiments.
  All methods receive the same B for each level/seed; fixed stripes remain the
  same across a video sequence. Results are invariant to batch/method/sample
  traversal order. A is evaluated once, not repeatedly averaged across seeds.
- Use `--pipelines m0 clahe` to evaluate only named configured chains.
- The evaluator defaults to **raw14** references, excluding uint8 mirror TIFFs
  and recording their IDs/counts. On validation all current frames are uint16;
  on test this policy selects 2,151 of 3,749 frames. Recompute A on that same
  subset: never compare its AP directly with the old full mixed-test AP.
- `--source-policy mixed` explicitly evaluates the mixed-bit-depth mirror as a
  separately labelled AGC experiment. Its source-aware normalization is fixed
  equally for A/B/C. A matching cache policy is required if using cached frames.
- PSNR is computed from global pixel-weighted MSE; SSIM is the frame mean with
  data range 255. Infinite PSNR is JSON null plus `psnr_infinite: true`.
- Timings separate reading, degradation, restoration, tensor conversion,
  inference and metric computation. First-use warm-up is included; checkpoint
  loading is excluded. Shared reading/degradation costs appear as pipeline costs
  in each condition and must not be summed as experiment wall time.

Choose parameters and method chains on **validation**. Run `--split test`
only after fixing them and use a different output directory. Mixed-bit-depth
and raw14 results must remain separate. The tools do not select a winning
method automatically or assume C improves on B.

For the first interpretable ablation, use `--levels light --degradation-steps
contrast` in the evaluator, or `--level light --degradation-steps contrast` in
the demo. The initial full-chain medium preset reached a detection floor in a
four-image smoke sample; this is not a full-dataset finding. Calibrate severity
and restoration parameters on validation instead of increasing detector epochs
or choosing settings from test.

### Current validation finding

The full 1,144-image validation run on 2026-10-04 used the current ResNet50
checkpoint, raw14 inputs, light contrast-only degradation and seed 0:

| Condition | mAP@0.5 | mAP@[0.5:0.95] |
|---|---:|---:|
| A: reference | 0.4963 | 0.2836 |
| B: degraded | 0.4741 | 0.2629 |
| C: m0 | 0.4741 | 0.2629 |
| C: clahe | 0.0475 | 0.0226 |
| C: median_clahe | 0.0248 | 0.0118 |
| C: column_median_clahe | 0.0211 | 0.0103 |

M0 reproduced B, but the configured CLAHE chains substantially reduced AP.
These are negative baseline results, not evidence that restoration improves
detection. They apply to this checkpoint and contrast-only validation experiment,
not every thermal video or degradation setting. Keep normalization and detector
weights fixed when testing alternatives. The local report is saved under
`outputs/detector/abc_val_contrast/`; it is intentionally not committed.

## Team quick start: model placement and running

Run all commands from the repository root. Each runnable example is one line
and can be pasted into PowerShell. This workflow uses the existing frozen model;
do not run `train_detector.py` merely to try the demo or reproduce evaluation.

### 1. Install and check the environment

Check out the same code revision as the checkpoint handoff, then install the
detector extra:

```powershell
uv sync --extra detector
```

Check device selection without starting training or inference:

```powershell
uv run --extra detector python scripts/check_cuda.py --device auto
```

Inspect `selected`, `cuda_available`, `device_name` and `cuda_error`. Auto mode
uses CUDA when usable and otherwise selects CPU; CPU execution is slower.
Use `--device cuda` if you want the command to fail instead of falling back.
The CUDA wheel installation alone does not guarantee a compatible GPU/driver.

### 2. Obtain and place the checkpoint

Ask the team maintainer for the approved checkpoint download location and its
matching code revision. No model download URL is bundled into this repository;
the dataset download script does not download the trained detector.

Keep the original filename and place the file here, relative to the repository:

```text
outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt
```

Create the destination folder if needed, then copy the downloaded `.pt` file
into it using your file manager. Do not replace an existing checkpoint without
checking which version it is.

```powershell
New-Item -ItemType Directory -Force -Path outputs/detector
```

Verify the exact filename and SHA256:

```powershell
Get-FileHash -LiteralPath outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt -Algorithm SHA256
```

For the current approved artifact (approximately 165.68 MiB), the expected hash
is:

```text
811af6e81679623ccd2a3fe85e9ab7f12bcb6332edc679b4fd93c0b6104dff17
```

Compare hashes case-insensitively. A mismatch means this is a different artifact;
check with the maintainer before comparing results. A future checkpoint needs
its own handoff hash. Only load trusted checkpoints: the current loader uses
`torch.load(..., weights_only=False)`, which may execute code in an untrusted file.

The checkpoint includes model weights, category mapping, model configuration
and thermal normalization statistics. Do not substitute a generic torchvision
weight file or override normalization to make enhanced images look better.
It is a frame-based detector, not a video-tracking model or a boat detector.

### 3. Place the input data

For a visual demo, a local grayscale thermal MP4 is enough; annotations and
training data are not required. Put your authorized demo video under
`data/test_inference/thermal_demo.mp4` (or use its actual path in `--source`).
An ordinary RGB webcam is not a substitute for a thermal camera.

For mAP evaluation, acquire the FLIR data using the project's authorized download
workflow. Keep the matching images and original COCO annotations together:

```text
thermal-video-restoration/
  outputs/
    detector/
      fasterrcnn_resnet50_thermal_norm_best.pt
  data/
    test_inference/
      thermal_demo.mp4                 # your own authorized demo input
    FLIR_ADAS_v2/
      images_thermal_val/
        coco.json
        analyticsData/
          *.tiff
      video_thermal_test/
        coco.json
        analyticsData/
          *.tiff
```

Evaluation requires only the split being evaluated, not the training split or a
training cache. The current mirror's test folder is approximately 996 MiB;
its `coco.json` is approximately 28 MiB and contains annotations, not images.
JSON files alone are insufficient. Do not rename TIFFs, remap category/image IDs
or randomly repartition frames. A small labelled subset is a smoke test, not
the full-split benchmark. Keep dataset redistribution permissions separate from
model sharing permissions; neither the data nor credentials belong in Git.

### 4. Run a visual detector demo

This example expects the demo filename from the layout above:

```powershell
uv run --extra detector python scripts/infer_realtime.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --source data/test_inference/thermal_demo.mp4 --allow-uint8 --device auto --score-threshold 0.3 --target-fps 30
```

Use `--allow-uint8` for 8-bit thermal MP4 input. This is detection on an AGC/video
reference, not a radiometric 14-bit evaluation. Space pauses/resumes; Q or Esc
quits. To check a machine without a graphical window, append
`--no-display --max-frames 20 --target-fps 0` instead of the existing FPS option.
No ground-truth mAP is available for an unlabelled video.

### 5. Reproduce the clean detector baseline

Full validation, reading raw TIFFs without a cache:

```powershell
uv run --extra detector python scripts/evaluate_detector.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --data-root data/FLIR_ADAS_v2 --split val --device auto --batch-size 2 --workers 2 --progress-interval 10 --output outputs/detector/team_resnet50_val.json
```

For the current artifact and full validation split, the recorded reference is
mAP@0.5 approximately 0.4963 and mAP@[0.5:0.95] approximately 0.2836. Small numerical
differences can occur across devices/batch sizes; compare checkpoint hash, data
scope and parameters first. On machines with limited memory, use `--workers 0`
and/or `--batch-size 1`.

For the existing mixed-bit-depth mirror test benchmark, use:

```powershell
uv run --extra detector python scripts/evaluate_detector.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --data-root data/FLIR_ADAS_v2 --split test --allow-uint8 --device auto --batch-size 2 --workers 2 --progress-interval 10 --output outputs/detector/team_resnet50_test_mixed.json
```

The recorded full mixed-test reference is mAP@0.5 approximately 0.2201 and
mAP@[0.5:0.95] approximately 0.1149 on 3,749 images. This is a different scope
from the raw14-only A/B/C test subset. Do not use test to choose restoration
parameters. Choose a fresh report filename for each run; the clean evaluator
can overwrite an existing `--output` file.

### 6. Run the A/B/C comparison on validation

Start with a four-image contrast-only smoke run, without a cache:

```powershell
uv run --extra detector python scripts/evaluate_abc.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --data-root data/FLIR_ADAS_v2 --split val --device auto --batch-size 2 --levels light --seeds 0 --degradation-steps contrast --max-images 4 --progress-interval 1 --output-dir outputs/detector/team_abc_smoke
```

Then run the full validation comparison:

```powershell
uv run --extra detector python scripts/evaluate_abc.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --data-root data/FLIR_ADAS_v2 --split val --device auto --batch-size 2 --levels light --seeds 0 --degradation-steps contrast --progress-interval 10 --output-dir outputs/detector/team_abc_val_contrast
```

The configured restoration chains are experimental; a completed run does not
mean C outperforms B. Read `report.md`, `metrics.csv` and `metrics.json` in the
output folder. The A/B/C output directory must be new or empty; use a different
name for a rerun. Omit `--degradation-steps contrast` to enable the full degradation
chain only when that is the intended experiment. A/B/C runs multiple detector
passes plus restoration/image metrics and takes longer than a clean evaluation.

### Troubleshooting and sharing checklist

- **Checkpoint not found:** verify the filename, `.pt` extension and repository
  working directory. `_archive/` contains old experiments, not the active model.
- **Missing torch/torchvision:** run `uv sync --extra detector` and keep
  `--extra detector` in the run command. Do not install packages into a different
  Python environment.
- **CUDA unavailable:** inspect `check_cuda.py` output and the machine's GPU/driver.
  `--device cpu` is a functional fallback, not a speed equivalent.
- **Missing TIFF/COCO file:** check the split layout; metadata alone is not enough.
- **Missing cache or policy mismatch:** omit `--cache-root` and use the original
  TIFFs. Do not share a cache as if it were the authoritative dataset.
- **No demo window/Tcl error:** use Python with Tcl/Tk for GUI mode, or
  `--no-display` for bounded checks. Do not install a second OpenCV package.
- **Existing A/B/C output:** select a new output directory instead of deleting
  previous experiment evidence.

Share code through Git. Share the approved checkpoint through a separate
team-approved artifact channel, with its SHA256, code revision and commands.
Team members obtain FLIR data through the authorized download workflow. Keep
`data/`, `outputs/`, `.pt` files, caches, recordings and credentials ignored;
do not force-add them to the repository. The README is tracked and shared with
the code, but the model file is not.
