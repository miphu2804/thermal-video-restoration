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
uv run python scripts/infer_realtime.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --source data/FLIR_ADAS_v2/video_thermal_test --sequence video-4FRnNpmSmwktFJKjg --device auto --compare-abc --pipeline column_median_clahe --level medium --seed 0 --score-threshold 0.3 --target-fps 30
```

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
uv run python scripts/evaluate_abc.py --checkpoint outputs/detector/fasterrcnn_resnet50_thermal_norm_best.pt --data-root data/FLIR_ADAS_v2 --split val --cache-root data/cache/detector_float32_v1 --device auto --batch-size 2 --levels medium --seeds 0 --progress-interval 10 --output-dir outputs/detector/abc_val
```

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
or choosing settings from test. The contrast-only smoke retained detections,
and M0 reproduced B; CLAHE still lowered AP in that tiny sample.
