"""Interactive frame-by-frame inference, not a benchmark or training script.

Uses Tk for the window because OpenCV is headless in this project. Throughput
is best-effort; slow inference slows playback rather than claiming 30 FPS.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from thermal_restore.data.experiment_config import (
    load_restoration_config,
    source_sequence,
)
from thermal_restore.data.stream import (
    iter_capture_frames,
    iter_tiff_frames,
    select_tiff_paths,
)
from thermal_restore.detection.device import require_device
from thermal_restore.detection.live import render_comparison, render_detections
from thermal_restore.detection.torchvision_backend import (
    TorchvisionDetector,
    frame_to_tensor,
    normalization_for_dtype,
)
from thermal_restore.detection.window import LiveWindow
from thermal_restore.experiments import build_restoration, degrade_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source",
        type=Path,
        help="A TIFF sequence directory, TIFF file, or local thermal video.",
    )
    source.add_argument(
        "--camera",
        type=int,
        help="OpenCV camera index; thermal AGC input expected, not RGB webcam.",
    )
    parser.add_argument(
        "--sequence",
        help="Required when the TIFF directory contains multiple FLIR sequences.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/FLIR_ADAS_v2/images_thermal_train/coco.json"),
        help="COCO file used only for category names; no labels are required.",
    )
    parser.add_argument(
        "--allow-uint8",
        action="store_true",
        help="Explicitly allow contrast-stretched TIFF or 8-bit AGC video/camera.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30,
        help="Playback cap, not a throughput promise; 0 runs as fast as possible.",
    )
    parser.add_argument("--max-frames", type=int, help="Optional smoke-run limit.")
    parser.add_argument(
        "--no-display", action="store_true", help="Headless diagnostic/export mode."
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Optional annotated .mp4; refuses to overwrite existing files.",
    )
    parser.add_argument(
        "--report", type=Path, help="Optional JSON timing/configuration report."
    )
    parser.add_argument(
        "--compare-abc",
        action="store_true",
        help="Show source, simulated degradation and restoration side by side.",
    )
    parser.add_argument(
        "--restoration-config", type=Path, default=Path("configs/abc_baselines.json")
    )
    parser.add_argument(
        "--pipeline",
        help="Configured restoration pipeline; uses config default when omitted.",
    )
    parser.add_argument(
        "--level", choices=("light", "medium", "heavy"), default="medium"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--degradation-steps",
        nargs="+",
        choices=("contrast", "blur", "fixed_stripes", "temporal_stripes", "noise"),
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=480,
        help="Display-only width of each A/B/C panel.",
    )
    args = parser.parse_args()
    if (
        not 0 <= args.score_threshold <= 1
        or not np.isfinite(args.target_fps)
        or args.target_fps < 0
        or (args.max_frames is not None and args.max_frames < 1)
        or (args.camera is not None and args.camera < 0)
        or args.seed < 0
        or args.panel_width < 1
    ):
        parser.error("Invalid threshold, target-fps, camera index or max-frames")
    if args.camera is not None and args.sequence:
        parser.error("--sequence applies only to TIFF directories")
    if args.output_video and (
        args.output_video.suffix.lower() != ".mp4" or args.output_video.exists()
    ):
        parser.error("--output-video must be a new .mp4 path")
    if args.report and (args.report.suffix.lower() != ".json" or args.report.exists()):
        parser.error("--report must be a new .json path")
    return args


def main():
    args = parse_args()
    comparison_config = None
    restore = None
    if args.compare_abc:
        comparison_config = load_restoration_config(args.restoration_config)
        args.pipeline = args.pipeline or comparison_config["default_pipeline"]
        if args.pipeline not in comparison_config["pipelines"]:
            raise SystemExit(f"Unknown pipeline: {args.pipeline}")
        restore = build_restoration(comparison_config["pipelines"][args.pipeline])
    if args.camera is not None:
        if not args.allow_uint8:
            raise SystemExit(
                "Video/camera requires --allow-uint8. RGB cameras are out of the thermal training domain."
            )
        frames = iter_capture_frames(args.camera, allow_uint8=True)
    elif args.source.is_dir() or args.source.suffix.lower() in {".tif", ".tiff"}:
        paths = select_tiff_paths(args.source, args.sequence)
        frames = iter_tiff_frames(paths, allow_uint8=args.allow_uint8)
        print(
            f"sequence={args.sequence or 'single source'} available_frames={len(paths)}"
        )
    else:
        if not args.source.is_file() or args.sequence or not args.allow_uint8:
            raise SystemExit(
                "Video requires an existing file, --allow-uint8, and no --sequence"
            )
        frames = iter_capture_frames(args.source, allow_uint8=True)
    import torch

    device = require_device(torch, args.device)
    print(f"device={device.selected} cuda={device.as_dict()}")
    detector = TorchvisionDetector.from_checkpoint(args.checkpoint, device.selected)
    detector.model.requires_grad_(False)
    if args.annotations.is_file():
        payload = json.loads(args.annotations.read_text(encoding="utf-8"))
        names = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    else:
        names = {}
        print("Category-name file unavailable; displaying original category IDs.")
    print(
        f"normalization={detector.input_normalization} mean={detector.input_mean} std={detector.input_std}"
    )
    print(
        "Display-only inference: not mAP evaluation or tracking. Restoration runs only with --compare-abc. FPS excludes model loading; first-frame warm-up is included."
    )
    if args.compare_abc:
        print(
            f"A/B/C demo: level={args.level} seed={args.seed} pipeline={args.pipeline}. One checkpoint and source-based normalization for all conditions. Three inference passes per source frame; no mAP without COCO labels."
        )
    if args.camera is not None or (
        args.source is not None
        and args.source.suffix.lower() not in {".tif", ".tiff"}
        and not args.source.is_dir()
    ):
        print(
            "AGC video is not raw thermal; color inputs become grayscale and accuracy is not guaranteed. Camera processing is sequential; driver buffering may add latency."
        )
    window = None
    writer = None
    count = 0
    inference_seconds = 0.0
    work_seconds = 0.0
    paused_seconds = 0.0
    recent = deque(maxlen=30)
    seen_dtypes = {}
    started = time.perf_counter()
    try:
        window = None if args.no_display else LiveWindow()
        while args.max_frames is None or count < args.max_frames:
            if window:
                window.pump()
                if window.closed:
                    break
                pause_started = time.perf_counter()
                while window.paused and not window.closed:
                    window.pump()
                    time.sleep(0.01)
                paused_seconds += time.perf_counter() - pause_started
                if window.closed:
                    break
            step_started = time.perf_counter()
            try:
                packet = next(frames)
            except StopIteration:
                break
            mean, std = normalization_for_dtype(
                packet.dtype,
                mean=detector.input_mean,
                std=detector.input_std,
                allow_uint8=args.allow_uint8,
            )
            seen_dtypes[str(packet.dtype)] = {"mean": mean, "std": std}
            variants = [packet.pixels]
            if args.compare_abc:
                sequence_key = args.sequence or (
                    source_sequence(Path(packet.label))
                    if args.source
                    and (
                        args.source.is_dir()
                        or args.source.suffix.lower() in {".tif", ".tiff"}
                    )
                    else str(args.source or args.camera)
                )
                degraded = degrade_frame(
                    packet.pixels,
                    sequence_key=sequence_key,
                    frame_key=Path(packet.label).stem,
                    level=args.level,
                    seed=args.seed,
                    steps=args.degradation_steps,
                )
                variants.extend([degraded, restore(degraded)])
            tensors = [frame_to_tensor(frame, mean=mean, std=std) for frame in variants]
            if detector.device.type == "cuda":
                torch.cuda.synchronize(detector.device)
            inference_started = time.perf_counter()
            condition_predictions = [
                detector.predict([tensor], [packet.index], args.score_threshold)
                for tensor in tensors
            ]
            predictions = condition_predictions[0]
            if detector.device.type == "cuda":
                torch.cuda.synchronize(detector.device)
            inference_elapsed = time.perf_counter() - inference_started
            inference_seconds += inference_elapsed
            recent.append(inference_elapsed)
            model_fps = len(recent) / sum(recent)
            active_elapsed = time.perf_counter() - started - paused_seconds
            display_fps = count / active_elapsed if count else 0.0
            if args.compare_abc:
                image = render_comparison(
                    variants,
                    condition_predictions,
                    names,
                    labels=[
                        f"A: source | boxes {len(condition_predictions[0])}",
                        f"B: {args.level} | boxes {len(condition_predictions[1])}",
                        f"C: {args.pipeline} | boxes {len(condition_predictions[2])}",
                    ],
                    panel_width=args.panel_width,
                    raw14=packet.dtype != np.dtype(np.uint8),
                )
                footer = np.zeros((30, image.shape[1], 3), dtype=np.uint8)
                cv2.putText(
                    footer,
                    f"{device.selected} | frame {packet.index} | A/B/C inference {inference_elapsed * 1000:.0f} ms | processing loop {display_fps:.1f} source FPS | seed {args.seed} | shared A display range",
                    (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                image = np.vstack([image, footer])
            else:
                image = render_detections(
                    packet.pixels,
                    predictions,
                    names,
                    status=f"{device.selected} | frame {packet.index} | model {model_fps:.1f} FPS | loop {display_fps:.1f} FPS | {inference_elapsed * 1000:.0f} ms | boxes {len(predictions)}",
                    stretch=packet.dtype != np.dtype(np.uint8),
                )
            if args.output_video:
                if writer is None:
                    args.output_video.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(args.output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.target_fps or 30,
                        (image.shape[1], image.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot open MP4 writer with mp4v codec")
                    output_shape = image.shape
                if image.shape != output_shape:
                    raise ValueError("Cannot record variable-size frames in one video")
                writer.write(image)
            if window:
                window.show(image)
            count += 1
            work_seconds += time.perf_counter() - step_started
            if count == 1 or count % 30 == 0:
                print(
                    f"frame={packet.index} processed={count} boxes={len(predictions)} model_fps={model_fps:.2f} inference_ms={inference_elapsed * 1000:.1f} dtype={packet.dtype} mean={mean} std={std}"
                )
            if args.target_fps:
                deadline = step_started + 1 / args.target_fps
                while time.perf_counter() < deadline:
                    if window:
                        window.pump()
                        if window.closed or window.paused:
                            break
                    time.sleep(min(0.005, max(0, deadline - time.perf_counter())))
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        frames.close()
        if writer:
            writer.release()
        if window:
            window.close()
    if not count:
        raise SystemExit("No frames were processed")
    elapsed = time.perf_counter() - started - paused_seconds
    report = {
        "checkpoint": str(args.checkpoint),
        "source": str(args.source) if args.source else f"camera:{args.camera}",
        "sequence": args.sequence,
        "device": device.as_dict(),
        "frames": count,
        "score_threshold": args.score_threshold,
        "allow_uint8": args.allow_uint8,
        "source_normalization": seen_dtypes,
        "target_fps": args.target_fps,
        "active_seconds": elapsed,
        "loop_fps": count / elapsed,
        "processing_fps": count / work_seconds,
        "model_fps": count / inference_seconds,
        "mean_inference_ms": inference_seconds / count * 1000,
        "timing_scope": "Includes first-frame warm-up; model loading excluded; loop includes pacing and rendering; camera may buffer.",
        "output_video": str(args.output_video) if args.output_video else None,
        "comparison": (
            {
                "level": args.level,
                "seed": args.seed,
                "steps": args.degradation_steps,
                "pipeline": args.pipeline,
                "stages": comparison_config["pipelines"][args.pipeline],
                "display_policy": "Shared A percentile range for raw TIFF; 0-255 for uint8; never used as model input",
                "inference_passes_per_source_frame": 3,
                "labelled_evaluation": False,
            }
            if args.compare_abc
            else None
        ),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
