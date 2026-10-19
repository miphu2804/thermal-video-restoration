"""Evaluate input-preprocessing candidates on a clean labelled FLIR split.

This script keeps one frozen detector and applies each candidate directly to the
same source frames. It answers whether a preprocessing transform improves
detection on validation data; it does not retrain the detector or simulate a
degradation. Choose a candidate on validation before reporting it on test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from thermal_restore.data import FrameCache, load_coco
from thermal_restore.data.experiment_config import load_restoration_config
from thermal_restore.data.loader import frame_dtype, load_frame
from thermal_restore.detection.device import require_device
from thermal_restore.detection.torchvision_backend import (
    TorchvisionDetector,
    frame_to_tensor,
    normalization_for_dtype,
)
from thermal_restore.experiments import (
    QualityTotals,
    build_restoration,
    range_remapping_stages,
)
from thermal_restore.metrics import mean_average_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("configs/inference_preprocessing.json"),
    )
    parser.add_argument("--pipelines", nargs="+", help="Pipelines; default: all")
    parser.add_argument("--source-policy", choices=("raw14", "mixed"), default="raw14")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/detector/preprocessing_val")
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or not 0 <= args.score_threshold <= 1
        or args.sample_seed < 0
        or args.progress_interval < 0
        or (args.max_images is not None and args.max_images < 1)
    ):
        parser.error(
            "Invalid batch size, score threshold, seed, progress interval or image limit"
        )
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error(
            "output-dir must be absent or empty; existing results are never overwritten"
        )
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state() -> dict:
    return {
        "predictions": [],
        "quality": QualityTotals(),
        "preprocessing_seconds": 0.0,
        "tensor_seconds": 0.0,
        "inference_seconds": 0.0,
        "quality_seconds": 0.0,
    }


def _predict(
    state: dict,
    frames: list[np.ndarray],
    references: list[np.ndarray],
    dtypes: list[np.dtype],
    image_ids: list[int],
    detector: TorchvisionDetector,
    args: argparse.Namespace,
    torch,
) -> None:
    started = time.perf_counter()
    tensors = []
    for frame, dtype in zip(frames, dtypes):
        mean, std = normalization_for_dtype(
            dtype,
            mean=detector.input_mean,
            std=detector.input_std,
            allow_uint8=args.source_policy == "mixed",
        )
        tensors.append(frame_to_tensor(frame, mean=mean, std=std))
    state["tensor_seconds"] += time.perf_counter() - started
    if detector.device.type == "cuda":
        torch.cuda.synchronize(detector.device)
    started = time.perf_counter()
    state["predictions"].extend(
        detector.predict(tensors, image_ids, args.score_threshold)
    )
    if detector.device.type == "cuda":
        torch.cuda.synchronize(detector.device)
    state["inference_seconds"] += time.perf_counter() - started
    started = time.perf_counter()
    for reference, frame in zip(references, frames):
        state["quality"].update(reference, frame)
    state["quality_seconds"] += time.perf_counter() - started


def _finish(state: dict, dataset, detector, condition: str, stages: list[dict], args):
    print(f"Computing mAP: {condition}...", flush=True)
    started = time.perf_counter()
    metrics = mean_average_precision(dataset, state["predictions"])
    map_seconds = time.perf_counter() - started
    quality = state["quality"].result()
    processing = (
        state["preprocessing_seconds"]
        + state["tensor_seconds"]
        + state["inference_seconds"]
    )
    row = {
        "condition": condition,
        "stages": stages,
        "images": len(dataset.images),
        "predictions": len(state["predictions"]),
        "map50": metrics.map50,
        "map50_95": metrics.map50_95,
        "ap50_by_category": {
            str(category_id): {
                "name": dataset.categories[category_id],
                "value": value,
            }
            for category_id, value in metrics.ap50_by_category.items()
        },
        "image_quality": quality,
        "timing": {
            "preprocessing_seconds": state["preprocessing_seconds"],
            "tensor_seconds": state["tensor_seconds"],
            "inference_seconds": state["inference_seconds"],
            "quality_seconds": state["quality_seconds"],
            "map_seconds": map_seconds,
            "processing_seconds_per_image": processing / len(dataset.images),
            "inference_seconds_per_image": state["inference_seconds"]
            / len(dataset.images),
        },
    }
    print(
        f"{condition} map50={row['map50']:.4f} map50_95={row['map50_95']:.4f} "
        f"mse={quality['mse']:.4f} ssim={quality['ssim']:.4f}"
    )
    return row


def main() -> None:
    args = parse_args()
    config = load_restoration_config(args.preprocessing_config)
    pipelines = args.pipelines or list(config["pipelines"])
    unknown = set(pipelines) - set(config["pipelines"])
    if unknown:
        raise SystemExit(f"Unknown pipelines: {sorted(unknown)}")
    if len(pipelines) != len(set(pipelines)):
        raise SystemExit("Duplicate pipelines are not allowed")
    restorers = {
        name: build_restoration(config["pipelines"][name]) for name in pipelines
    }

    folder = args.data_root / (
        "images_thermal_val" if args.split == "val" else "video_thermal_test"
    )
    original = load_coco(folder / "coco.json", folder)
    dtypes = {
        image.image_id: frame_dtype(image.path)
        for image in tqdm(
            original.images,
            desc="inspect source policy",
            unit="image",
            disable=args.progress_interval == 0,
        )
    }
    unsupported = {
        str(dtype)
        for dtype in dtypes.values()
        if dtype not in (np.dtype(np.uint16), np.dtype(np.uint8))
    }
    if unsupported:
        raise SystemExit(f"Unsupported TIFF source dtypes: {unsupported}")
    eligible = [
        image
        for image in original.images
        if args.source_policy == "mixed"
        or dtypes[image.image_id] == np.dtype(np.uint16)
    ]
    if args.max_images is not None and args.max_images < len(eligible):
        indices = np.sort(
            np.random.default_rng(args.sample_seed).choice(
                len(eligible), args.max_images, replace=False
            )
        )
        eligible = [eligible[int(index)] for index in indices]
    if not eligible:
        raise SystemExit("No eligible source frames")
    selected_ids = {image.image_id for image in eligible}
    dataset = type(original)(
        images=tuple(eligible),
        annotations=tuple(
            annotation
            for annotation in original.annotations
            if annotation.image_id in selected_ids
        ),
        categories=dict(original.categories),
    )
    allow_uint8 = args.source_policy == "mixed"
    cache = (
        FrameCache(args.cache_root / args.split, original, allow_uint8=allow_uint8)
        if args.cache_root
        else None
    )
    cache_indices = {
        image.image_id: index for index, image in enumerate(original.images)
    }

    import torch

    device = require_device(torch, args.device)
    detector = TorchvisionDetector.from_checkpoint(args.checkpoint, device.selected)
    detector.model.eval()
    detector.model.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = {"A": _state(), **{f"P:{name}": _state() for name in pipelines}}
    decode_seconds = 0.0
    started = time.perf_counter()
    batches = range(0, len(dataset.images), args.batch_size)
    progress = tqdm(
        batches,
        total=(len(dataset.images) + args.batch_size - 1) // args.batch_size,
        desc="preprocessing candidates",
        unit="batch",
        dynamic_ncols=True,
        disable=args.progress_interval == 0,
    )
    for batch_number, start in enumerate(progress, 1):
        images = dataset.images[start : start + args.batch_size]
        ids = [image.image_id for image in images]
        dtypes_batch = [dtypes[image.image_id] for image in images]
        load_started = time.perf_counter()
        references = [
            (
                cache.load(cache_indices[image.image_id])
                if cache
                else load_frame(image.path, allow_uint8=allow_uint8)
            )
            for image in images
        ]
        decode_seconds += time.perf_counter() - load_started
        for image, frame in zip(images, references):
            if frame.shape != (image.height, image.width):
                raise ValueError(f"TIFF/COCO dimension mismatch: {image.path}")
        _predict(
            states["A"],
            references,
            references,
            dtypes_batch,
            ids,
            detector,
            args,
            torch,
        )
        for name, restore in restorers.items():
            transform_started = time.perf_counter()
            processed = [restore(frame) for frame in references]
            states[f"P:{name}"]["preprocessing_seconds"] += (
                time.perf_counter() - transform_started
            )
            _predict(
                states[f"P:{name}"],
                processed,
                references,
                dtypes_batch,
                ids,
                detector,
                args,
                torch,
            )
        if args.progress_interval and batch_number % args.progress_interval == 0:
            progress.set_postfix(
                images=min(start + args.batch_size, len(dataset.images))
            )

    rows = [_finish(states["A"], dataset, detector, "A", [], args)]
    for name in pipelines:
        row = _finish(
            states[f"P:{name}"],
            dataset,
            detector,
            f"P:{name}",
            config["pipelines"][name],
            args,
        )
        row["range_remapping_stages"] = range_remapping_stages(
            config["pipelines"][name]
        )
        row["delta_map50_vs_A"] = row["map50"] - rows[0]["map50"]
        row["delta_map50_95_vs_A"] = row["map50_95"] - rows[0]["map50_95"]
        rows.append(row)

    scope = (
        "smoke_sample" if args.max_images is not None else "full_source_policy_subset"
    )
    selected_dtypes = Counter(str(dtypes[image.image_id]) for image in dataset.images)
    summary = {
        "version": 1,
        "split": args.split,
        "evaluation_scope": scope,
        "source_policy": args.source_policy,
        "full_split_images": len(original.images),
        "images": len(dataset.images),
        "image_ids": list(dataset.image_ids),
        "annotation_ids": [item.annotation_id for item in dataset.annotations],
        "dtype_counts": dict(selected_dtypes),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "annotations_path": str(folder / "coco.json"),
        "annotations_sha256": _sha256(folder / "coco.json"),
        "model_category_ids": list(detector.category_ids),
        "input_normalization": detector.input_normalization,
        "device": device.as_dict(),
        "batch_size": args.batch_size,
        "score_threshold": args.score_threshold,
        "preprocessing_config": config,
        "pipelines": pipelines,
        "sample_seed": args.sample_seed,
        "max_images": args.max_images,
        "decode_seconds": decode_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "conditions": rows,
        "notes": [
            "A is the clean source reference; P candidates are applied directly to the same source frame.",
            "The detector checkpoint and source-specific normalization are identical for every condition.",
            "Parameters must be selected on validation before a test run.",
            "Image quality is measured against the original source, not absolute thermal ground truth.",
        ],
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open(
        "x", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "condition",
            "images",
            "map50",
            "map50_95",
            "delta_map50_vs_A",
            "delta_map50_95_vs_A",
            "mse",
            "psnr_db",
            "ssim",
            "processing_seconds_per_image",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(
                        key,
                        row["image_quality"].get(key, row["timing"].get(key)),
                    )
                    for key in fields
                }
            )
    lines = [
        "# Frozen detector preprocessing comparison",
        "",
        f"Scope: **{scope}**, split: **{args.split}**, source policy: **{args.source_policy}**.",
        "",
        "| Condition | mAP@0.5 | mAP@[0.5:0.95] | Delta mAP@0.5 vs A | MSE | PSNR (dB) | SSIM | Processing ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        quality = row["image_quality"]
        psnr = "inf" if quality["psnr_infinite"] else f"{quality['psnr_db']:.2f}"
        delta = row.get("delta_map50_vs_A")
        lines.append(
            f"| {row['condition']} | {row['map50']:.4f} | {row['map50_95']:.4f} | {delta:+.4f} | {quality['mse']:.3f} | {psnr} | {quality['ssim']:.4f} | {row['timing']['processing_seconds_per_image'] * 1000:.1f} |"
            if delta is not None
            else f"| {row['condition']} | {row['map50']:.4f} | {row['map50_95']:.4f} | - | {quality['mse']:.3f} | {psnr} | {quality['ssim']:.4f} | {row['timing']['processing_seconds_per_image'] * 1000:.1f} |"
        )
    lines.extend(
        [
            "",
            "A preprocessing candidate is useful only if it improves validation detection without an unacceptable cost or image-quality regression. No automatic winner is selected.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
