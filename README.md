# thermal-video-restoration

Restoring degraded thermal infrared video before object detection (IMP302m). Proposal (Vietnamese): [docs/proposal.md](docs/proposal.md).

## Setup

```bash
uv sync
uv run pre-commit install
scripts/download_dataset.sh   # FLIR ADAS v2 thermal frames (~4.7 GB) into data/
```

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
configs/        # experiment YAML configs
notebooks/      # exploration and dataset checks
scripts/        # dataset download
tests/
data/           # local data, not committed (FLIR license forbids redistribution)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the git workflow.

## Interactive thermal detector inference

This project studies restoration of degraded thermal video, then measures
whether a fixed detector recovers detection quality. The live demo below is
**detector-only**, not restoration, tracking, or a replacement for mAP evaluation.
It loads your existing checkpoint without retraining or downloading weights.

Run from the repository root (one line):

```powershell
uv run python scripts/infer_realtime.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --source data/FLIR_ADAS_v2/video_thermal_test --sequence video-4FRnNpmSmwktFJKjg --device auto --score-threshold 0.3 --target-fps 30
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
